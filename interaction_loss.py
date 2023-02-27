import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset import params
from model import get_model
from DWT import *
from PIL import Image
import os
class BaseAttack(object):
    def __init__(self, attack_name, model_name, target):
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
    def _show_images(self, inps):
        unnorm_inps = self._mul_std_add_mean(inps)


        image = unnorm_inps[0].permute([1,2,0]) # c,h,w to h,w,c
        image[image<0] = 0
        image[image>1] = 1
        image = Image.fromarray((image.detach().cpu().numpy()*255).astype(np.uint8))
        # print ('Saving to ', save_path)
        image.show()

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
class InteractionLoss(nn.Module):

    def __init__(self, target=None, label=None):
        super(InteractionLoss, self).__init__()
        assert (target is not None) and (label is not None)
        self.target = target
        self.label = label
    def _show_images(self, inps):
        unnorm_inps = self._mul_std_add_mean(inps)


        image = unnorm_inps[0].permute([1,2,0]) # c,h,w to h,w,c
        image[image<0] = 0
        image[image>1] = 1
        image = Image.fromarray((image.detach().cpu().numpy()*255).astype(np.uint8))
        # print ('Saving to ', save_path)
        image.show()

    def logits_interaction(self, outputs_adv, leave_one_outputs,
                           only_add_one_outputs, zero_outputs):
        complete_score = outputs_adv[:, self.target] - outputs_adv[:, self.label]
        leave_one_out_score = (
            leave_one_outputs[:, self.target] -
            leave_one_outputs[:, self.label])

        only_add_one_score = (
            only_add_one_outputs[:, self.target] -
            only_add_one_outputs[:, self.label])
        zero_score = (
            zero_outputs[:, self.target] - zero_outputs[:, self.label])
        pairwise_interaction = complete_score - leave_one_out_score - only_add_one_score +zero_score
        average_pairwise_interaction = (complete_score - leave_one_out_score -
                                        only_add_one_score +
                                        zero_score).mean()
        # the ir loss little is better
        test4 =torch.abs(complete_score - leave_one_out_score -only_add_one_score + zero_score)
        test4[test4>=torch.abs(average_pairwise_interaction)] = 0

        return average_pairwise_interaction,test4,pairwise_interaction

    def forward(self, outputs, leave_one_outputs, only_add_one_outputs,
                zero_outputs):
        return self.logits_interaction(outputs, leave_one_outputs,
                                       only_add_one_outputs, zero_outputs)


def sample_grids(sample_grid_num=16,
                 grid_scale=16,
                 img_size=224,
                 sample_times=8):
    grid_size = img_size // grid_scale
    sample = []
    sample_times = sample_times//2

    for _ in range(sample_times):
        grids = []
        ids = np.random.randint(0, grid_scale ** 2, size=sample_grid_num)
        rows, cols = ids // grid_scale, ids % grid_scale
        for r, c in zip(rows, cols):
            grid_range = (slice(r * grid_size * 2, (r + 1) * grid_size * 2),
                          slice(c * grid_size * 2, (c + 1) * grid_size * 2))
            grids.append(grid_range)
        sample.append(grids)
    for _ in range(sample_times):
        grids = []
        ids = np.random.randint(0, grid_scale ** 2, size=sample_grid_num)
        rows, cols = ids // grid_scale, ids % grid_scale
        for r, c in zip(rows, cols):
            grid_range = (slice(r * grid_size  * 3, (r + 1) * grid_size * 3),
                          slice(c * grid_size * 3, (c + 1) * grid_size * 3))
            grids.append(grid_range)
        sample.append(grids)
    return sample
def _show_images( inps):
    #unnorm_inps = self._mul_std_add_mean(inps)


    image = inps[0].permute([1,2,0]) # c,h,w to h,w,c
    image[image<0] = 0
    image[image>1] = 1
    image = Image.fromarray((image.detach().cpu().numpy()*255).astype(np.uint8))
    # print ('Saving to ', save_path)
    image.show()
    image.close()

def sample_for_interaction(
                           adv_high_mask,
                           only_add_one_mask_vit,
                           delta,
                           sample_grid_num,
                           grid_scale,
                           img_size,
                           times=16,

                           ):
    '''
    samples = sample_grids(
        sample_grid_num=sample_grid_num,
        grid_scale=grid_scale,
        img_size=img_size,
        sample_times=times//3 )
    '''
    #bundary_mean=torch.mean(abs(abs_add_perturbation1))
    bundary_mean_abs=adv_high_mask.mean()

    only_add_one_mask = torch.zeros_like(delta).repeat(times , 1, 1, 1)
    count = times //3
    count1 = times //2
    adv_high_mask_pre = torch.clone(adv_high_mask)
    #abs_add_perturbation1_pre = torch.clone(abs_add_perturbation1)
    #te = abs_add_perturbation1[0]
    '''
    for i in range(times // 3):
        if bundary_mean == 0:

            only_add_one_mask[i: i + 1, :, :, :] = only_add_one_mask_vit[i:i+1,:,:,:]
        if bundary_mean != 0:
            bundary_mean = (1 +i/12)*bundary_mean
            abs_add_perturbation1[abs_add_perturbation1_pre <= bundary_mean] = 0
            abs_add_perturbation1[abs_add_perturbation1_pre > bundary_mean] = 1
            only_add_one_mask[i: i + 1 , :, :, :] = abs_add_perturbation1
            #te = abs_add_perturbation1[i]
            #_show_images(abs_add_perturbation1[0])
            #abs_add_perturbation1_pre = torch.clone(abs_add_perturbation1)
    '''
    #only_add_one_mask[0: 10 , :, :, :] = only_add_one_mask_vit_attention
    for j in range(times // 2):
        bounday_high_mask = (1 +j/2) * bundary_mean_abs
        adv_high_mask_pre[adv_high_mask_pre <= bounday_high_mask] = 0
        adv_high_mask_pre[adv_high_mask_pre > bounday_high_mask] = 1
        only_add_one_mask[j: j + 1, :, :, :] = adv_high_mask_pre
        #_show_images(adv_high_mask_pre)
        adv_high_mask_pre = torch.clone(adv_high_mask)
    only_add_one_mask[count1 : count1+15, :, :, :] = only_add_one_mask_vit
    '''
    for i in range(times//3):
        grids = samples[i]
        for grid in grids:
            only_add_one_mask[i:i + 1, :, grid[0], grid[1]] = 1
            #b_test = only_add_one_mask[i:i + 1, 0:1, :, :].cpu().numpy()
        #a_test  = only_add_one_mask[i:i + 1, :, :, :].cpu().numpy()
    '''
    leave_one_mask = 1 - only_add_one_mask
    only_add_one_perturbation = delta * only_add_one_mask
    leave_one_out_perturbation = delta * leave_one_mask

    return only_add_one_perturbation, leave_one_out_perturbation,only_add_one_mask,leave_one_mask


transform_BZ = transforms.Normalize(
        mean=[0.4850, 0.4560, 0.4060],  # 取决于数据集
        std=[0.2290, 0.2240, 0.2250]
    )
transform_compose = transforms.Compose([

        transform_BZ
    ])
def get_features(
    model,
    x,
    perturbation,
    leave_one_out_perturbation,
    only_add_one_perturbation,
):

    #iaozhunhua
    '''
    img_transform = transform_compose(x + perturbation)

    outputs = model(img_transform)
    x_leave_one_out_perturbation = transform_compose(x + leave_one_out_perturbation)
    leave_one_outputs = model(x_leave_one_out_perturbation)
    x_only_add_one_perturbation = transform_compose(x + only_add_one_perturbation)
    only_add_one_outputs = model(x_only_add_one_perturbation)
    x_zero = transform_compose(x)
    zero_outputs = model(x_zero)
    '''
    outputs = model( perturbation)

    leave_one_outputs = model(leave_one_out_perturbation)

    only_add_one_outputs = model(only_add_one_perturbation)

    zero_outputs = model(x)

    return (outputs, leave_one_outputs, only_add_one_outputs, zero_outputs)
