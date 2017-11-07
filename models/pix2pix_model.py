import numpy as np
import torch
import os
from collections import OrderedDict
from torch import autograd
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.loss_G_GAN = Variable(torch.FloatTensor([0]).cuda())
        # self.loss_G_L1 = Variable(torch.FloatTensor([0]).cuda())
        # self.loss_D = Variable(torch.FloatTensor([0]).cuda())
        self.loss_D_with_gp = Variable(torch.FloatTensor([0]).cuda())

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # use_sigmoid = False
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=opt.lr)
            self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        # print('self.fake_B', self.fake_B.size())
        self.real_B = Variable(self.input_B)

    def forward_with_noise(self):
        self.real_A = Variable(self.input_A)
        batch_size = self.real_A.size(0)
        fixed_noise = Variable(torch.FloatTensor(batch_size, 3, 256, 256).normal_(0, 1).cuda())
        real_A_with_noise = torch.cat((self.real_A, fixed_noise), 1)
        self.fake_B = self.netG.forward(real_A_with_noise)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def get_gradient_penalty(self):
        batch_size, c, h, w = self.real_B.size()

        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, c*h*w).contiguous().view(
            batch_size, c, h, w)
        alpha = alpha.cuda()

        real_B_plus_fake_B = alpha * self.real_B.data + ((1 - alpha) * self.fake_B.data)
        real_AB_plus_fake_B = torch.cat((self.real_A.data, real_B_plus_fake_B), 1)
        real_AB_plus_fake_B = Variable(real_AB_plus_fake_B, requires_grad=True)
        loss_D_real_plus_fake = self.netD(real_AB_plus_fake_B)
        gradients = autograd.grad(outputs=loss_D_real_plus_fake, inputs=real_AB_plus_fake_B,
                                  grad_outputs=torch.ones(loss_D_real_plus_fake.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True
                                  )
        gradients = gradients[0]
        lambda_gp = 10  # hyperparameter: Gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        return gradient_penalty

    def backward_wgan_D(self):
        # real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        loss_D_real = self.netD.forward(real_AB)
        loss_D_real = loss_D_real.view(-1, 1).mean(0)
        # The Numbers 1, -1
        one = torch.FloatTensor([1]).cuda()
        mone = one * -1
        loss_D_real.backward(mone)

        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        loss_D_fake = self.netD.forward(fake_AB.detach())
        loss_D_fake = loss_D_fake.view(-1, 1).mean(0)
        loss_D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = self.get_gradient_penalty()
        gradient_penalty.backward()
        # print "gradien_penalty: ", gradient_penalty

        self.loss_D_with_gp = loss_D_real - loss_D_fake + gradient_penalty
        self.loss_D = loss_D_real - loss_D_fake

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def backward_wgan_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.loss_G_GAN = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.loss_G_GAN.view(-1, 1).mean(0)
        self.loss_G_GAN = -self.loss_G_GAN
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # self.loss_G = self.loss_G_GAN
        # The Numbers -1
        mone = -1 * torch.FloatTensor([1]).cuda()
        self.loss_G_GAN.backward(mone)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_netD_parameters(self):
        make_trainable(self.netD, True)
        for p in self.netD.parameters():
            p.data.clamp(-0.01, 0.01)
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_wgan_D()
        self.optimizer_D.step()

    def optimize_netD_parameters_gp(self):
        make_trainable(self.netD, True)

        self.forward()
        self.optimizer_D.zero_grad()

        self.backward_wgan_D()

        self.optimizer_D.step()

    def optimize_netG_parameters(self):
        make_trainable(self.netD, False)
        # self.forward()
        self.optimizer_G.zero_grad()
        self.backward_wgan_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            # ('D_real', self.loss_D_real.data[0]),
                            # ('D_fake', self.loss_D_fake.data[0]),
                            # ('D_GAN', self.loss_D.data[0]),
                            ('D_GAN_with_gp', self.loss_D_with_gp.data[0]),
                            # ('G_L1', self.loss_G_L1.data[0]),
                            ]
                           )

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def step_D(self, v, init_grad):
        err = self.netD(v)
        err.backward(init_grad)
        return err


def make_trainable(net, val):
    for p in net.parameters():
        p.requires_grad = val
