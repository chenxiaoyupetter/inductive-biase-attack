import torch
import argparse
from torch.utils.data import DataLoader
import os
import pandas as pd
import time

from dataset import AdvDataset
from model import get_model
from utils import BASE_ADV_PATH, accuracy, AverageMeter
#0.453 vit-small   0.751 cia s24
def arg_parse():
    '''
    'vit_base_patch16_224',
    'pit_b_224',
    'cait_s24_224',
    'visformer_small',
    '''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--adv_path1', type=str,default='/home/car/PNA-PatchOut-master-yuanshi/paper_results/model_vit_base_patch16_224-method_OurAlgorithm-paper_results', help='the path of adversarial examples.')
    parser.add_argument('--adv_path2', type=str,default='/home/car/PNA-PatchOut-master-yuanshi/paper_results/model_pit_b_224-method_OurAlgorithm-paper_results', help='the path of adversarial examples.')
    parser.add_argument('--adv_path3', type=str,default='/home/car/PNA-PatchOut-master-yuanshi/paper_results/model_cait_s24_224-method_OurAlgorithm-paper_results', help='the path of adversarial examples.')
    parser.add_argument('--adv_path4', type=str, default='/home/car/PNA-PatchOut-master-yuanshi/paper_results/model_visformer_small-method_OurAlgorithm-paper_results', help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='input batch size for reference (default: 16)')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--boundary', type=float, default='', help='')
    args = parser.parse_args()
    args.adv_path1 = os.path.join( '{}{}'.format(args.adv_path1,str(args.boundary)))
    args.adv_path2 = os.path.join('{}{}'.format(args.adv_path2,str(args.boundary)))
    args.adv_path3 = os.path.join('{}{}'.format(args.adv_path3,str(args.boundary)))
    args.adv_path4 = os.path.join( '{}{}'.format(args.adv_path4,str(args.boundary)))
    return args

if __name__ == '__main__':
    args = arg_parse()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    df_asr = pd.DataFrame(columns=[

        'vit_base_patch16_224',
        'pit_b_224',
        'cait_s24_224',
        'visformer_small',
        'deit_base_distilled_patch16_224',
        'tnt_s_patch16_224',
        'levit_256',
        'convit_base',
        'inception_v3',
        'inception_v4',
        'inception_resnet_v2',
        'resnetv2_152',
        'ens_adv_inception_resnet_v2',



                                   ])
    # Loading dataset
    df_asr_ = pd.DataFrame(columns=[

        'filename',
        'label',
    ])
    model_name_lt = [
        'vit_base_patch16_224',
        'pit_b_224',
        'cait_s24_224',
        'visformer_small',
        'deit_base_distilled_patch16_224',
        'tnt_s_patch16_224',
        'levit_256',
        'convit_base',
        'inception_v3',
        'inception_v4',
        'inception_resnet_v2',
        'resnetv2_152',
        'ens_adv_inception_resnet_v2',


                                   ]
    '''
    'vit_base_patch16_224',
        'pit_b_224',
        'cait_s24_224',
        'visformer_small',
        'deit_base_distilled_patch16_224',
        'convit_base',
        'inception_v3',
        'inception_v4',
        'inception_resnet_v2',
    '''
    for model_name in model_name_lt:

        asr = []

        for i in range(4):

            if i ==0:
                dataset = AdvDataset(model_name, args.adv_path1)
            if i ==1:
                dataset = AdvDataset(model_name, args.adv_path2)
            if i ==2:
                dataset = AdvDataset(model_name, args.adv_path3)
            if i ==3:
                dataset = AdvDataset(model_name, args.adv_path4)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            '''
            query_paths = data_loader.dataset.query_paths
            json_info = data_loader.dataset.json_info
            filename = []
            label = []

            for query_paths_ in query_paths:
                id_ = json_info[query_paths_]['class_id']
                query_paths_ = query_paths_.replace('JPEG','png')
                filename.append(query_paths_)
                label.append(id_)
            df_asr_['filename'] = filename
            df_asr_['label'] = label
            df_asr_.to_csv(os.path.join('/home/car/PNA-PatchOut-master-yuanshi/',
                                       'ps-{}.csv'.format(str(args.boundary))),
                          index=False)
        '''




            #print (test)

            model = get_model(model_name)
            model.cuda()
            model.eval()

            # main
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()

            prediction = []
            gts = []
            with torch.no_grad():

                end = time.time()
                for batch_idx, batch_data in enumerate(data_loader):
                    if batch_idx%10 == 0:
                        print ('Ruing batch_idx', batch_idx)
                    batch_x = batch_data[0].cuda()
                    batch_y = batch_data[1].cuda()
                    batch_name = batch_data[2]

                    output = model(batch_x)
                    acc1, acc5 = accuracy(output.detach(), batch_y, topk=(1, 5))
                    top1.update(acc1.item(), batch_x.size(0))
                    top5.update(acc5.item(), batch_x.size(0))

                    batch_time.update(time.time() - end)
                    end = time.time()

                    _, pred = output.detach().topk(1, 1, True, True)
                    pred = pred.t()
                    prediction += list(torch.squeeze(pred.cpu()).numpy())
                    gts += list(batch_y.cpu().numpy())
                test = 100-top1.avg
                asr.append(test)
        df_asr[model_name] = asr
        '''
        df = pd.DataFrame(columns = ['path', 'pre', 'gt'])
        df['path'] = dataset.paths[:len(prediction)]
        df['pre'] = prediction
        df['gt'] = gts

        j = 0
        for i in range(1000):
            if prediction[i] == gts[i]:
                j = j+1
        avg = j/1000
        print(avg)
        print(1-avg)
        asr = 1-avg


        #df_asr.to_csv()
        print("avg",top1.avg)
        '''
    #df.to_csv(os.path.join(args.adv_path, 'prediction-model_{}-top1_{:.3f}.csv'.format(args.model_name, top1.avg)), index=False)
    df_asr.to_csv(os.path.join('/home/car/PNA-PatchOut-master-yuanshi/', 'prediction-model-mi-{}.csv'.format(str(args.boundary))),
              index=False)