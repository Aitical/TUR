import torch
from torch import nn as nn
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn import functional as F
import math
from copy import deepcopy
import os
import copy
import random
#from memory_profiler import profile

from basicsr.archs import build_network
from basicsr.losses import build_loss,l1_loss_type
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import torch
import numpy as np


@MODEL_REGISTRY.register()
class AIRnet_uncertainty(BaseModel):   #纯净原版
    """Base IR model, it will recored 7 type of losses."""

    def __init__(self, opt):
        super(AIRnet_uncertainty, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            #self.net_g.load_state_dict(torch.load(load_path, map_location=torch.device(self.device)),strict=False)

        if self.is_train:
            self.init_training_settings()

        self.loss_noise_15=torch.Tensor([100])[0].to(self.device)
        self.loss_noise_25=torch.Tensor([100])[0].to(self.device)
        self.loss_noise_50=torch.Tensor([100])[0].to(self.device)
        self.loss_derain=torch.Tensor([100])[0].to(self.device)
        self.loss_dehaze=torch.Tensor([100])[0].to(self.device)
        #self.loss_dark=torch.Tensor([100])[0].to(self.device)
        #self.loss_sr=torch.Tensor([100])[0].to(self.device)
        # self.old_loss_dict = [torch.tensor([0], device=self.device) for i in range(7)]
        # self.uncertainty_loss_dict = [torch.tensor([0], device=self.device) for i in range(7)]
        # self.uncertainty_bn_dict = [torch.tensor([0], device=self.device) for i in range(7)]
        
        
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            #net =  build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                # self.net_g_ema.load_state_dict(torch.load(load_path, map_location=torch.device(self.device)),strict=False)
                # checkpoint = torch.load(load_path, map_location=torch.device(self.device))
                # model_state_dict = self.net_g_ema.state_dict()
                # filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                # model_state_dict.update(filtered_checkpoint)
                # self.net_g_ema.load_state_dict(model_state_dict, strict=False)
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
                # load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
                # for k, v in deepcopy(load_net).items():
                #     if k.startswith('module.'):
                #         load_net[k[7:]] = v
                #         load_net.pop(k)
                # crt_net = self.get_bare_model(net)
                # crt_net = self.get_bare_model(crt_net)
                # crt_net = crt_net.state_dict()
                # crt_net_keys = set(crt_net.keys())
                # load_net_keys = set(load_net.keys())

                # logger = get_root_logger()
                # if crt_net_keys != load_net_keys:
                #     logger.warning('Current net - loaded net:')
                #     for v in sorted(list(crt_net_keys - load_net_keys)):
                #         logger.warning(f'  {v}')
                #     logger.warning('Loaded net - current net:')
                #     for v in sorted(list(load_net_keys - crt_net_keys)):
                #         logger.warning(f'  {v}')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pixtype = build_loss(train_opt['pixel_opt']).to(self.device)
            #self.cri_pixtype = uncertainty_loss().to(self.device)#l1_loss_type().to(self.device)
            # self.cri_pixtype_dwa = l1_loss_type_DWA().to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        if self.is_train:
            self.typeword=data['img_type_word']
            self.lq = data['lq'].to(self.device)
            #self.lq2 = data['lq2'].to(self.device)
         
            self.gt = data['gt'].to(self.device)
            #self.gt2 = data['gt2'].to(self.device)
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output,self.uncertanty = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            # l_pix = self.cri_pix(self.output, self.gt)
            # l_total += l_pix
            if current_iter %1 ==0:
                # return total_loss/5,category_losses,old_loss_dict,uncertainty_bn_dict
                l_total,loss_list,old_loss_dict,uncertainty_bn_dict,uncertainty_loss_dict=self.cri_pixtype(self.output, self.gt,self.typeword,self.uncertanty)
                loss_dict['l_pix'] = l_total
                if loss_list[0] != 0:
                    self.loss_noise_15 = loss_list[0].detach()

                if loss_list[1] != 0:
                    self.loss_noise_25 = loss_list[1].detach()

                if loss_list[2] != 0:
                    self.loss_noise_50 = loss_list[2].detach()

                if loss_list[3] != 0:
                    self.loss_derain = loss_list[3].detach()

                if loss_list[4] != 0:
                    self.loss_dehaze = loss_list[4].detach()

                

                loss_dict['loss_noise_15'] = self.loss_noise_15
                loss_dict['loss_noise_25'] = self.loss_noise_25
                loss_dict['loss_noise_50'] = self.loss_noise_50
                loss_dict['loss_derain'] = self.loss_derain
                loss_dict['loss_dehaze'] = self.loss_dehaze
                old_loss_name = ['old_loss_noise_15','old_loss_noise_25','old_loss_noise_50','old_loss_derain','old_loss_dehaze']
                uncertainty_loss_name = ['uncertainty_loss_noise_15','uncertainty_loss_noise_25','uncertainty_loss_noise_50','uncertainty_loss_derain','uncertainty_loss_dehaze']
                uncertainty_bn_name = ['uncertainty_bn_noise_15','uncertainty_bn_noise_25','uncertainty_bn_noise_50','uncertainty_bn_derain','uncertainty_bn_dehaze']
                i = 0
                for name in old_loss_name:
                    loss_dict[name]=old_loss_dict[i]
                    i+=1
                i = 0
                for name in uncertainty_loss_name:
                    loss_dict[name]=uncertainty_loss_dict[i]
                    i+=1
                i=0
                for name in uncertainty_bn_name:
                    loss_dict[name]=uncertainty_bn_dict[i]
                    i+=1
    
                # print('test')
                # print(loss_dict)
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

   
    
    def test(self):

        self.net_g.eval()
        with torch.no_grad():
            #self.output = self.net_g(self.lq,self.lq)
            self.output,self.uncertanty = self.net_g(self.lq,self.lq)
        self.net_g.train()

    def tile_test(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """

        _, _, H_ori, W_ori = self.lq.shape

        window_size = 128
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        self.tile_size = 128
        self.tile_pad = 0
        self.scale = 1

        self.img = img
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                # try:
                with torch.no_grad():
                    output_tile = self.net_g([input_tile,self.img_typee.to(self.device)])
                # except RuntimeError as error:
                #     print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]

        self.output = self.output[:, :, :H_ori, :W_ori]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train=False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader): 
            #print(val_data)
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            #self.tile_test()
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
                # if '001' in save_img_path or '0801' in save_img_path or 'haze' in save_img_path:
                #     imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1) #取消显示测试图片名称
            #pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.is_train=True

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
        
        
        
