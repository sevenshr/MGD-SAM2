import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import ttach as tta


to_pil = transforms.ToPILImage()

def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, save_dir=None,if_save=False):
    
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8 = 'f1', 'auc', 'none', 'none', 'none', 'none', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8 = 'f_mea', 'mae', 'none', 'none', 'none', 'none', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8 = 'shadow', 'non_shadow', 'ber', 'none', 'none', 'none', 'none', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod_p
        metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8 = 'maxf', 'wfm', 'em', 'sm', 'mae', 'none', 'none', 'none'
    elif eval_type == 'kvasir':
        metric_fn = utils.calc_kvasir
        metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8 = 'dice', 'iou', 'none', 'none', 'none', 'none', 'none', 'none'

    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1,1.25], interpolation='bilinear', align_corners=False),

    ]
    )

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    val_metric5 = utils.Averager()
    val_metric6 = utils.Averager2()
    val_metric7 = utils.Averager2()
    val_metric8 = utils.Averager2()

    pbar = tqdm(loader, leave=False, desc='val')
    
    cnt = 0
    with torch.no_grad():
        for batch in pbar:
            for k, v in batch.items():
                if k != 'im_path':
                    batch[k] = v.cuda()

            image_name = batch['im_path'][0].split('/')[-1].split('.')[0]
            f_name = batch['im_path'][0].split('/')[-3]

            inp = batch['inp']
            inp_glb = batch['inp_glb']

            pred = []
            for transformer in transforms:  
                rgb_trans = transformer.augment_image(inp)
                rgb_trans_glb = transformer.augment_image(inp_glb)

                model.inp_size = rgb_trans.shape[-1] //2
                model._bb_feat_sizes = [
            (int(model.inp_size/4),int(model.inp_size/4)),
            (int(model.inp_size/8),int(model.inp_size/8)),
            (int(model.inp_size/16),int(model.inp_size/16)),
                                            ]

                model.image_embedding_size = model.inp_size // model.patch_size

                model_output = model.infer(rgb_trans)

                deaug_mask = transformer.deaugment_mask(model_output)
                pred.append(deaug_mask)
            
            pred = torch.mean(torch.stack(pred, dim=0), dim=0)

            pred = F.interpolate(pred, batch['gt'].shape[-2:], mode="bilinear", align_corners=False)
            pred = torch.sigmoid(pred)


            if if_save:
                prediction = to_pil(pred.data.squeeze().cpu())

                if cnt == 0:
                    save_dir_ = os.path.join(save_dir,f_name)
                    os.makedirs(save_dir_, exist_ok=True)
                    save_path = os.path.join(save_dir_, f"{image_name}.png")
                else:
                    save_path = os.path.join(save_dir,f_name, f"{image_name}.png")
                prediction.save(save_path)

            result1, result2, result3, result4, result5, result6, result7, result8 = metric_fn(pred, batch['gt'])

            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            try:
                val_metric3.add(result3.item(), inp.shape[0])
                val_metric4.add(result4.item(), inp.shape[0])
                val_metric5.add(result5.item(), inp.shape[0])
                val_metric6.add(result6, inp.shape[0])
                val_metric7.add(result7, inp.shape[0])
                val_metric8.add(result8, inp.shape[0])
            except:
                pass

            if verbose:
                pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
                pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
                try:
                    pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
                    pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))
                    pbar.set_description('val {} {:.4f}'.format(metric5, val_metric5.item()))
                except:
                    pass
            cnt += 1
    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_metric5.item(), val_metric6.item(), val_metric7.item(), val_metric8.item()


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    parser.add_argument('--save-dir', default='./prediction')
    parser.add_argument('--if-save', default=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8, shuffle=False)

    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)

    metric1, metric2, metric3, metric4, metric5, metric6, metric7, metric8 = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True,save_dir=args.save_dir,if_save=args.if_save)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
    print('metric5: {:.4f}'.format(metric5))
    config_basename = os.path.basename(args.config)
    config_name, _ = os.path.splitext(config_basename)
    suffix = config_name.split('_')[-1]
    output_txt = f"./output_{suffix}.txt"
    with open(output_txt, "w") as f:

        f.write('metric1: {:.4f}\n'.format(metric1))
        f.write('metric2: {:.4f}\n'.format(metric2))
        f.write('metric3: {:.4f}\n'.format(metric3))
        f.write('metric4: {:.4f}\n'.format(metric4))
        f.write('metric5: {:.4f}\n'.format(metric5))
        f.write('metric6: {}\n'.format(np.array2string(metric6, separator=', ')))
        f.write('metric7:{}\n'.format(np.array2string(metric7, separator=', ')))
        f.write('metric8:{}\n'.format(np.array2string(metric8, separator=', ')))


