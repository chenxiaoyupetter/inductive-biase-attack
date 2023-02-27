import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
import scipy.stats as st
import copy
from utils import ROOT_PATH
from functools import partial
import copy
import pickle as pkl
from torch.autograd import Variable
import torch.nn.functional as F
import spam
from dataset import params
from model import get_model
from DWT import *
import matplotlib.pyplot as plt

from interaction_loss import (InteractionLoss, get_features,
                               sample_for_interaction)
class BaseAttack(object):
    def __init__(self, attack_name, model_name, target, decay=1.0):
        self.attack_name = attack_name
        self.model_name = model_name
        self.target = target
        if self.target:
            self.loss_flag = -1
        else:
            self.loss_flag = 1
        self.used_params = params(self.model_name)

        # loading model
        self.model = get_model(self.model_name)
        self.model.cuda()
        self.model.eval()

        self.decay = decay
    def forward(self, *input):
        """
        Rewrite
        """
        raise NotImplementedError

    def _mul_std_add_mean(self, inps):
        dtype = inps.dtype
        #print("self.used_params['mean']",self.used_params['mean'])
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        inps.mul_(std[:,None, None]).add_(mean[:,None,None])
        return inps

    def _sub_mean_div_std(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        inps.sub_(mean[:,None,None]).div_(std[:,None,None])
        return inps

    def _save_images(self, inps, filenames, output_dir):
        unnorm_inps = self._mul_std_add_mean(inps)
        for i,filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            image = unnorm_inps[i].permute([1,2,0]) # c,h,w to h,w,c
            image[image<0] = 0
            image[image>1] = 1
            image = Image.fromarray((image.detach().cpu().numpy()*255).astype(np.uint8))
            # print ('Saving to ', save_path)
            image.save(save_path)
    def _show_images(self, inps1):
        #unnorm_inps1 = self._mul_std_add_mean(inps1)


        image2 = inps1[0].permute([1,2,0]) # c,h,w to h,w,c
        image2 = image2[:,:,1]
        image2[image2<=0] = 0
        image2[image2>=1] = 1
        image2 = Image.fromarray((image2.detach().cpu().numpy()*255).astype(np.uint8))
        #image_color = image.shape()[0]
        # print ('Saving to ', save_path)

        #plt.imshow(image2, cmap='gray')
        plt.imshow(image2)
        plt.show()

        #image.show()

    def _update_inps(self, inps, grad, step_size):
        unnorm_inps = self._mul_std_add_mean(inps.clone().detach())
        unnorm_inps = unnorm_inps + step_size * grad.sign()
        unnorm_inps = torch.clamp(unnorm_inps, min=0, max=1).detach()
        adv_inps = self._sub_mean_div_std(unnorm_inps)
        return adv_inps

    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts

    def _return_perts(self, clean_inps, inps):
        clean_unnorm = self._mul_std_add_mean(clean_inps.clone().detach())
        adv_unnorm = self._mul_std_add_mean(inps.clone().detach())
        return adv_unnorm - clean_unnorm

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images

class OurAlgorithm(BaseAttack):
    def __init__(self, model_name, bundary,ablation_study='1,1,1', sample_num_batches=130, lamb=0.1, steps=10, epsilon=16/255, target=False):
        super(OurAlgorithm, self).__init__('OurAlgorithm', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.device = 'cuda:0'
        self.ablation_study = ablation_study.split(',')
        self.lamb = lamb
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches
        self.total_variation = spam.TotalVariation()
        wave: str = 'haar'
        self.DWT = DWT_2D_tiny(wavename=wave)
        self.IDWT = IDWT_2D_tiny(wavename=wave)
        self.bundary = bundary

        if self.ablation_study[2] == '1':
            print ('Using Skip')
            self._register_model()
        else:
            print ('Not Using Skip')


    def _register_model(self):
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
                for i in range(12):
                    self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)

    def _show_images(self,inps):
        # unnorm_inps = self._mul_std_add_mean(inps)

        image = inps[0].permute([1, 2, 0])  # c,h,w to h,w,c
        image[image < 0] = 0
        image[image > 1] = 1
        image = Image.fromarray((image.detach().cpu().numpy() * 255).astype(np.uint8))
        # print ('Saving to ', save_path)
        image.show()
        image.close()
    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        add_noise_mask_1 = torch.zeros_like(perts)
        times = 15
        only_add_one_mask_vit = torch.zeros_like(perts).repeat(times, 1, 1, 1)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
            #self._show_images(add_noise_mask)
        add_perturbation = perts * add_noise_mask
        for j in range(times):
            random.shuffle(ids)
            random.seed(seed)
            random.shuffle(ids)
            ids = np.array(ids[:self.sample_num_batches])
            rows, cols = ids // grid_num_axis, ids % grid_num_axis
            flag = 0
            for r1, c1 in zip(rows, cols):


                if flag<110:
                    add_noise_mask_1[:, :, r1 * self.crop_length:(r1 + 1) * self.crop_length,
                    c1 * self.crop_length:(c1 + 1) * self.crop_length] = 1
                    flag = flag + 1
            only_add_one_mask_vit[j:j+1,:,:,:] = add_noise_mask_1
            #self._show_images(add_noise_mask_1)
            add_noise_mask_1 = torch.zeros_like(perts)

        return add_perturbation, only_add_one_mask_vit

    def dis_spam(self,x, y):
        l2 = torch.sqrt(torch.sum((x - y) ** 2))
        '''
        manhattan = torch.sum(np.abs(x - y))
        chebyshev = torch.max(np.abs(x - y))

        featuremat = torch.mat([x, y])
        mv1 = torch.mean(featuremat[0])
        mv2 = torch.mean(featuremat[1])
        # 计算两列标准差
        dv1 = torch.std(featuremat[0])
        dv2 = torch.std(featuremat[1])
        # 相关系数和相关距离
        corrf = torch.mean(torch.multiply(featuremat[0] - mv1, featuremat[1] - mv2) / (dv1 * dv2))

        from scipy.spatial.distance import pdist

        # print(pdist([x, y], "correlation") ,corrf) # 相关距离
        # print(np.corrcoef([x, y]))  # 相关系数

        #print(l2, manhattan, chebyshev, corrf)
        '''
        return l2

    def forward(self, inps, labels):
        #print("inps",inps)
        #inps_ori = torch.squeeze(inps)
        #inps_numpy = inps_ori.numpy()
        lowFre_loss = nn.SmoothL1Loss(reduction='sum')
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()
        #inputs_ll = self.DWT(inps)
        #inputs_ll = self.IDWT(inputs_ll)
        #self._show_images(inputs_ll)
        unnorm_inps = self._mul_std_add_mean(inps)
        inputs_ll = self.DWT(unnorm_inps)
        # self._show_images(inputs_ll)
        momentum = torch.zeros_like(inps).cuda()

        inputs_ll = self.IDWT(inputs_ll)
        #te = unnorm_inps - inps
        perts = torch.zeros_like(unnorm_inps).cuda()
        #perts is adversarial
        perts.requires_grad_()
        #spam_inps_ori =  self.total_variation(unnorm_inps, 3).cuda()
        #spam_inps_ori.requires_grad_()
        #grad_pre =torch.zeros_like(unnorm_inps).cuda()
        #add_perturbation_pre = torch.zeros_like(unnorm_inps).cuda().requires_grad_()
        #ir_attention = torch.zeros_like(unnorm_inps).cuda()
        dandu = 0
        for i in range(self.steps):
            if self.ablation_study[0] == '1':

                print ('Using Pathes')
                add_perturbation,only_add_one_mask_vit = self._generate_samples_for_interactions(perts, i)
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
                #outputs11 = self.model(unnorm_inps )
                #a1= unnorm_inps
            else:
                print ('Not Using Pathes')
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts)))

            if self.ablation_study[1] == '1':
                print ('Using L2')
                cost1 = self.loss_flag * loss(outputs, labels).cuda()
                cost2 = torch.norm(perts)
                danceng_only_add_one_mask_vit = only_add_one_mask_vit[0]
                #inps_ori = torch.squeeze(inps)
                #image = inps_ori.permute([1, 2, 0])  # c,h,w to h,w,c

                add_perturbation1 = copy.deepcopy(add_perturbation.detach())
                #grad_pre = add_perturbation1
                #add_perturbation1 = grad_pre + add_perturbation1
                #add_perturbation1_mask = torch.ones_like(add_perturbation1)
                #add_perturbation1_01 = self._mul_std_add_mean(add_perturbation1)
                #add_perturbation1_01[add_perturbation1_01<=0.5]=0
                #add_perturbation1_01[add_perturbation1_01 > 0.5] = 1.5
                '''
                only_add_one_mask_vit_attention = torch.clone(only_add_one_mask_vit)
                for boundar_num in range(10):
                    bundary1 = torch.mean(abs(add_perturbation1))*(1+boundar_num/5)
                    #bundary = torch.median(abs(add_perturbation1))

                    abs_add_perturbation1 = abs(add_perturbation1)
                    abs_add_perturbation1[abs_add_perturbation1 <= bundary1] = 0
                    abs_add_perturbation1[abs_add_perturbation1 > bundary1] = 1
                    self._show_images(abs_add_perturbation1)
                    only_add_one_mask_vit_attention[boundar_num: boundar_num + 1 , :, :, :] = abs_add_perturbation1
                '''
                #mask = abs_add_perturbation1*add_perturbation1_mask
                #add_perturbation = add_perturbation + add_perturbation_pre * abs_add_perturbation1
                #add_perturbation_pre = copy.deepcopy(add_perturbation.detach())

                adv_ll = self.DWT(unnorm_inps + add_perturbation)
                adv_ll = self.IDWT(adv_ll)
                adv_pri = unnorm_inps
                adv_high_mask = abs(adv_ll - adv_pri)
                # adv_high_mask = adv_high_mask[:, :, 0]




                self.grid_scale = 16
                self.sample_times = 30
                self.sample_grid_num = 32
                self.image_width = 224
                #outputs2 = self.model(unnorm_inps + add_perturbation)
                only_add_one_perturbation, leave_one_out_perturbation,only_add_one_mask,leave_one_mask = \
                    sample_for_interaction(adv_high_mask,only_add_one_mask_vit,perts, self.sample_grid_num,
                                           self.grid_scale, self.image_width,
                                           self.sample_times)
                unnorm_inps_ir = copy.deepcopy(unnorm_inps.detach())
                #unnorm_inps_ir = self._sub_mean_div_std(unnorm_inps_ir)
                #unnorm_inps means the orginal image
                ir_out = self._sub_mean_div_std(unnorm_inps + add_perturbation)
                leave_one_out_perturbation_ir = self._sub_mean_div_std(unnorm_inps + leave_one_out_perturbation)
                only_add_one_perturbation_ir = self._sub_mean_div_std(unnorm_inps + only_add_one_perturbation)

                (outputs_adv, leave_one_outputs, only_add_one_outputs,
                 zero_outputs) = get_features(self.model, unnorm_inps_ir,ir_out,
                                              leave_one_out_perturbation_ir,
                                              only_add_one_perturbation_ir)
                outputs1_c = copy.deepcopy(outputs_adv.detach())
                #test_out = outputs1_c-outputs
                int_label = int(labels)
                outputs1_c[:, int_label] = -np.inf
                other_max = outputs1_c.max(1)[1].item()
                interaction_loss = InteractionLoss(
                    target=other_max, label=int_label)
                average_pairwise_interaction,ir_area,pairwise_interaction = interaction_loss(
                    outputs_adv, leave_one_outputs, only_add_one_outputs,
                    zero_outputs)
                #index_area = torch.nonzero(ir_area)
                #ir_area__numpy = ir_area.numpy()
                #index_area = np.argwhere(ir_area__numpy > 0)
                cost4 = -average_pairwise_interaction





                #t= inps_numpy.transpose(1,2,0)
                #print(t.shape)
                #t_image = np.floor(255*t)

                #spam_inps_adv = self.total_variation(unnorm_inps + add_perturbation, 3).cuda()
                #adv_image = adv.numpy()
                #adv_image_rgb = np.floor(255*adv_image)
                #l2_adv = spam.spam_extract_2(adv_image_rgb, 3)
                #cost5 = lowFre_loss(spam_inps_adv, spam_inps_ori).cuda()
                #cost3 = torch.nn.functional.mse_loss(spam_inps_ori,spam_inps_adv)
                #cost3 = torch.norm(spam_inps_adv)

                #print(bounday_high_mask,adv_high_mask)
                #self._show_images(abs_add_perturbation1)
                #print(abs_add_perturbation1)
                #self._show_images(adv_pri)
                #self._show_images(adv_ll)
                #self._show_images(adv_high_mask)
                #self._show_images(20 * adv_high_mask)


                cost3 = lowFre_loss(adv_ll, inputs_ll)
                high_mask = abs(inputs_ll - unnorm_inps)
                cost5 = lowFre_loss(high_mask, adv_high_mask)

                cost = cost1 + self.lamb * cost2 + 0.8 * cost4 + self.bundary/(cost5+0.1) +self.bundary*cost3
                #cost = cost4

                #print("cos4",cost4,average_pairwise_interaction)
            else:
                print ('Not Using L2')
                cost = self.loss_flag * loss(outputs, labels).cuda()

            cost.backward()
            grad = perts.grad.data
            '''
            grad1 = copy.deepcopy(grad.detach())
            abs_grads = abs(grad1)
            sum_grads = torch.sum(abs_grads)
            count_grads = torch.count_nonzero(abs_grads)
            bundary_grads = sum_grads/count_grads
            abs_grads[abs_grads <= bundary_grads] = 0
            abs_grads[abs_grads > bundary_grads] = self.bundary
            '''
            dandu_now = -pairwise_interaction[0]
            if dandu_now > dandu:
                pass
            #for location in index_area:
                #ir_attention = only_add_one_mask[location,:,:,:]+ir_attention
            #ir_attention_numpy = ir_attention.cpu().numpy()
            grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
            #grad += momentum * self.decay
            #momentum = grad


            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
            #st = self._sub_mean_div_std(unnorm_inps+perts.data)
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None,(self._sub_mean_div_std(10*perts.data)).detach()

class OurAlgorithm_MI(BaseAttack):
    def __init__(self, model_name, bundary, ablation_study='0,0,0', sample_num_batches=130, lamb=0.1, steps=10, epsilon=16/255, target=False, decay=1.0):
        super(OurAlgorithm_MI, self).__init__('OurAlgorithm_MI', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.decay = decay
        self.ablation_study = ablation_study.split(',')
        self.lamb = lamb
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches

        if self.ablation_study[2] == '1':
            print ('Using Skip')
            self._register_model()
        else:
            print ('Not Using Skip')
    
    def _register_model(self):   
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
                for i in range(12):
                    self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(inps).cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()

        for i in range(self.steps):
            if self.ablation_study[0] == '1':
                print ('Using Pathes')
                add_perturbation = self._generate_samples_for_interactions(perts, i)
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
            else:
                print ('Not Using Pathes')
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts)))

            if self.ablation_study[1] == '1':
                print ('Using L2')
                cost1 = self.loss_flag * loss(outputs, labels).cuda()
                cost2 = torch.norm(perts)
                cost = cost1 + self.lamb * cost2
            else:
                print ('Not Using L2')
                cost = self.loss_flag * loss(outputs, labels).cuda()
            cost.backward()
            grad = perts.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=[1,2,3], keepdim=True)
            grad += momentum*self.decay
            momentum = grad
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None,(self._sub_mean_div_std(10*perts.data)).detach()

class OurAlgorithm_SGM(BaseAttack):
    def __init__(self, model_name,  bundary,sgm_control='1,0', sample_num_batches=130, lamb=0.1, steps=10, epsilon=16/255, target=False):
        super(OurAlgorithm_SGM, self).__init__('OurAlgorithm_SGM', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.sgm_control = sgm_control.split(',')

        self.lamb = lamb
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches

        self._register_model()

    def _register_model(self):   
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        def mlp_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0], grad_in[1])
        
        def attn_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0], grad_in[1])

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
        mlp_hook_func = partial(mlp_mask_grad, gamma=0.5)
        attn_hook_func = partial(attn_mask_grad, gamma=0.5)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
                if self.sgm_control[0] == '1':
                    self.model.blocks[i].mlp.register_backward_hook(mlp_hook_func)
                if self.sgm_control[1] == '1':
                    self.model.blocks[i].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                if self.sgm_control[0] == '1':
                    self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_hook_func)
                if self.sgm_control[1] == '1':
                    self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.blocks[block_ind].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.blocks[block_ind].attn.qkv.register_backward_hook(attn_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.blocks_token_only[block_ind-24].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.blocks_token_only[block_ind-24].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.stage2[block_ind].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.stage2[block_ind].attn.qkv.register_backward_hook(attn_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.stage3[block_ind-4].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':           
                        self.model.stage3[block_ind-4].attn.qkv.register_backward_hook(attn_hook_func)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()

        for i in range(self.steps):
            perts.requires_grad_()
            add_perturbation = self._generate_samples_for_interactions(perts, i)
            outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
            cost1 = self.loss_flag * loss(outputs, labels).cuda()
            cost2 = torch.norm(perts)

            cost = cost1 + self.lamb * cost2
            cost.backward()
            grad = perts.grad.data
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None,(self._sub_mean_div_std(10*perts.data)).detach()

class OurAlgorithm_SGM_MI(BaseAttack):
    def __init__(self, model_name,bundary, sgm_control='1,0', sample_num_batches=130, lamb=0.1, steps=10, epsilon=16/255, target=False, decay=1.0):
        super(OurAlgorithm_SGM_MI, self).__init__('OurAlgorithm_SGM_MI', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.sgm_control = sgm_control.split(',')
        self.decay = decay

        self.lamb = lamb
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches
        self.bundary = bundary
        self._register_model()

    def _register_model(self):   
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        def mlp_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0], grad_in[1])
        
        def attn_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0], grad_in[1])

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
        mlp_hook_func = partial(mlp_mask_grad, gamma=0.5)
        attn_hook_func = partial(attn_mask_grad, gamma=0.5)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
                if self.sgm_control[0] == '1':
                    self.model.blocks[i].mlp.register_backward_hook(mlp_hook_func)
                if self.sgm_control[1] == '1':
                    self.model.blocks[i].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                if self.sgm_control[0] == '1':
                    self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_hook_func)
                if self.sgm_control[1] == '1':
                    self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.blocks[block_ind].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.blocks[block_ind].attn.qkv.register_backward_hook(attn_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.blocks_token_only[block_ind-24].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.blocks_token_only[block_ind-24].attn.qkv.register_backward_hook(attn_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.stage2[block_ind].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':
                        self.model.stage2[block_ind].attn.qkv.register_backward_hook(attn_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)
                    if self.sgm_control[0] == '1':
                        self.model.stage3[block_ind-4].mlp.register_backward_hook(mlp_hook_func)
                    elif self.sgm_control[1] == '1':           
                        self.model.stage3[block_ind-4].attn.qkv.register_backward_hook(attn_hook_func)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        #print("innnn")
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(inps).cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()

        for i in range(self.steps):
            perts.requires_grad_()
            add_perturbation = self._generate_samples_for_interactions(perts, i)
            outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
            cost1 = self.loss_flag * loss(outputs, labels).cuda()
            cost2 = torch.norm(perts)

            cost = cost1 + self.lamb * cost2
            cost.backward()
            grad = perts.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=[1,2,3], keepdim=True)
            grad += momentum*self.decay
            momentum = grad
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None,(self._sub_mean_div_std(10*perts.data)).detach()