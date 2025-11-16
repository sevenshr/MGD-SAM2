import argparse
import os
from functools import partial
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset
import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
import numpy as np
import random
import logging
from collections import OrderedDict

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
            
def adjust_learning_rate(optimizer, lr, index_split=-1, scale_lr=0.1):
    for index in range(len(optimizer.param_groups)):
        if index <= index_split:
            optimizer.param_groups[index]['lr'] = lr * scale_lr
        else:
            optimizer.param_groups[index]['lr'] = lr 
    return lr

def adjust_learning_rate_poly(optimizer, learning_rate, i_iter, max_iter, power, freeze_maskdecoder=False):
    split = 0 if freeze_maskdecoder else -1
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    lr = adjust_learning_rate(optimizer, lr, index_split=split)
    return lr

def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    if tag != 'train':
        dataset = datasets.make(spec['dataset'])
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    else:
        if 'dataset1' not in spec:  # 代表只有一组训练集 
            dataset = datasets.make(spec['dataset'])

        elif 'dataset2' not in spec:  # 代表只有两组训练集 0， 1

            dataset = datasets.make(spec['dataset'])  # 数据地址等一串信息
            dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})  # 添加到 wrapper 中
            dataset1 = datasets.make(spec['dataset1'])
            dataset1 = datasets.make(spec['wrapper1'], args={'dataset': dataset1})
            
            dataset = ConcatDataset([dataset, dataset1])

    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))
    if tag=='train':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True, sampler=sampler,drop_last=True)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = False)
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True, sampler=sampler,drop_last=False)
    return loader



def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader1 = make_data_loader(config.get('val_dataset'), tag='val')
    val_loader2 = make_data_loader(config.get('val_dataset2'), tag='val')
    val_loader3 = make_data_loader(config.get('val_dataset3'), tag='val')
    val_loader4 = make_data_loader(config.get('val_dataset4'), tag='val')
    val_loader5 = make_data_loader(config.get('val_dataset5'), tag='val')
    val_loader = [val_loader1,val_loader2,val_loader3,val_loader4,val_loader5]
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'kvasir':
        metric_fn = utils.calc_kvasir
        metric1, metric2, metric3, metric4 = 'dice', 'iou', 'none', 'none'
        

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    
    val_metric1 = 0
    val_metric2 = 0
    val_metric3 = 0
    val_metric4 = 0
    cnt = 0
    
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        with torch.no_grad():
            pred = torch.sigmoid(model.infer(inp))


        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]
        
        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1 += (result1 * pred.shape[0])
        val_metric2 += (result2 * pred.shape[0])
        val_metric3 += (result3 * pred.shape[0])
        val_metric4 += (result4 * pred.shape[0])     
        cnt += pred.shape[0]
        if pbar is not None:
            pbar.update(1)
    val_metric1 = torch.tensor(val_metric1).cuda()
    val_metric2 = torch.tensor(val_metric2).cuda()
    val_metric3 = torch.tensor(val_metric3).cuda()
    val_metric4 = torch.tensor(val_metric4).cuda()
    cnt = torch.tensor(cnt).cuda()
    dist.all_reduce(val_metric1)
    dist.all_reduce(val_metric2)
    dist.all_reduce(val_metric3)
    dist.all_reduce(val_metric4)
    dist.all_reduce(cnt)
          
    if pbar is not None:
        pbar.close()
    
    return val_metric1.item()/cnt, val_metric2.item()/cnt, val_metric3.item()/cnt, val_metric4.item()/cnt, metric1, metric2, metric3, metric4


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, epoch_start


def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    loss_list_aux = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']

        gt = batch['gt']
        model.set_input(inp, gt)

        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        batch_loss_aux = [torch.zeros_like(model.loss_G_aux) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss_aux, model.loss_G_aux)
        loss_list_aux.extend(batch_loss_aux)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    loss_aux = [i.item() for i in loss_list_aux]
    return mean(loss), mean(loss_aux)


def main(config_, save_path, args):
    
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader_list = make_data_loaders()
    
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, epoch_start = prepare_training()
   
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    state_dict = sam_checkpoint['model']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():

        if "iou_prediction_head"  in k or "mask_tokens" in k:
            print(k)
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)

    lr_1 = []
    lr_0_1 = []

    for name, para in model.named_parameters():
        if "image_encoder" in name:
            if "trunk" in name:
                if "mpadapter" in name or "new" in name:
                    lr_1.append(para)
                else:
                    para.requires_grad_(False)

            elif "neck" in name:
                lr_0_1.append(para)

            else:
                lr_1.append(para)

        elif "mask_decoder" in name:
            if "mask_tokens" in name or "new" in name:

                lr_1.append(para)

            else:
                lr_0_1.append(para)

        else:
            lr_1.append(para)


    params = [{'params': lr_0_1, 'lr': config['optimizer']['args']['lr'] * 0.1, 'weight_decay': 1e-5},
        {'params': lr_1, 'lr': config['optimizer']['args']['lr'], 'weight_decay': 0}]
    
   
    model.optimizer = utils.make_optimizer(
            params, config['optimizer'])

    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

        
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    max_val_v = [max_val_v] * len(val_loader_list)

    max_val_v_all = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()

        adjust_learning_rate_poly(model.optimizer, config['optimizer']['args']['lr'], epoch-1, epoch_max, power=0.9, freeze_maskdecoder=True)
        train_loss_G,train_loss_G_aux = train(train_loader, model)


        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', model.optimizer.param_groups[0]['lr'], epoch)
            log_info.append('lr: lr={:.8f}'.format(model.optimizer.param_groups[0]['lr']))
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)
            log_info.append('train G: loss_aux={:.4f}'.format(train_loss_G_aux))
            log_info.append('train G: loss_main={:.4f}'.format(train_loss_G-train_loss_G_aux))
            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = model.optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1_dict = []
            result2_dict = []
            for index, val_loader in enumerate(val_loader_list):
                result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
                    eval_type=config.get('eval_type'))

                if local_rank == 0:
                    log_info.append('val_{}: {}={:.4f}'.format(index, metric1, result1))
                    writer.add_scalars(metric1, {'val_{}'.format(index): result1}, epoch)
                    log_info.append('val_{}: {}={:.4f}'.format(index, metric2, result2))
                    writer.add_scalars(metric2, {'val_{}'.format(index): result2}, epoch)
                    log_info.append('val_{}: {}={:.4f}'.format(index, metric3, result3))
                    writer.add_scalars(metric3, {'val_{}'.format(index): result3}, epoch)
                    log_info.append('val_{}: {}={:.4f}'.format(index, metric4, result4))
                    writer.add_scalars(metric4, {'val_{}'.format(index): result4}, epoch)

                    result1_dict.append(result1)
                    result2_dict.append(result2)

                    if config['eval_type'] != 'ber':
                        if result1 > max_val_v[index]:
                            max_val_v[index] = result1
                            save(config, model, save_path, 'best_{}'.format(index))
                    else:
                        if result2 < max_val_v[index]:
                            max_val_v[index] = result2
                            save(config, model, save_path, 'best_{}'.format(index))

                    t = timer.t()
                    prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                    t_epoch = utils.time_text(t - t_epoch_start)
                    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                    log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
            
            if local_rank ==0:
                result1_avg = sum(result1_dict) / len(result1_dict)
                result2_avg = sum(result2_dict) / len(result2_dict)
                log_info.append('val_all: {}={:.4f}'.format(metric1, result1_avg))
                log_info.append('val_all: {}={:.4f}'.format(metric2, result2_avg))
                log(', '.join(log_info))
                writer.flush()
                
                
                if config['eval_type'] != 'ber':
                    if result1_avg > max_val_v_all:
                        max_val_v_all = result1
                        save(config, model, save_path, 'best')
                else:
                    if result2_avg < max_val_v_all:
                        max_val_v_all = result2
                        save(config, model, save_path, 'best')
            
            del result1, result2, result3, result4, metric1, metric2, metric3, metric4
            torch.cuda.empty_cache()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            mpadapter = model.encoder.backbone.mpadapter.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": mpadapter, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
