import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']
#############
# 7任务使用的损失函数
#############

@LOSS_REGISTRY.register()
class uncertainty_loss_softplus(nn.Module):
    
    def __init__(self):
        super(uncertainty_loss_softplus, self).__init__()
        self.device = torch.device('cuda')

    def forward(self, pred, target, types ,uncertainty):
        old_loss_dict = [torch.tensor(0, device=pred.device) for i in range(7)]
        uncertainty_loss_dict = [torch.tensor(0, device=pred.device) for i in range(7)]
        uncertainty_bn_dict = [torch.tensor(0, device=pred.device) for i in range(7)]
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list = []
        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list = []
        blur_list_ = []
        noise_list_ = []
        jpeg_list_ = []
        rain_list_ = []
        haze_list_ = []
        dark_list_ = []
        sr_list_ = []
        blur_lq_list_ = []
        noise_lq_list_ = []
        jpeg_lq_list_ = []
        rain_lq_list_ = []
        haze_lq_list_ = []
        dark_lq_list_ = []
        sr_lq_list_ = []
        blur_un_list = []
        noise_un_list = []
        jpeg_un_list = []
        rain_un_list = []
        haze_un_list = []
        dark_un_list = []
        sr_un_list = []
        # #elu = nn.Softplus()
        # for i in range(len(uncertainty)):
        #     uncertainty[i] = F.softplus(uncertainty[i])
            
        for p, t, y ,img_type in zip(pred, target,uncertainty ,types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
                rain_un_list.append(y)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
                haze_un_list.append(y)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
                blur_un_list.append(y)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
                noise_un_list.append(y)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
                jpeg_un_list.append(y)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
                dark_un_list.append(y)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)
                sr_un_list.append(y)
        
        num_task = 0
        
        blur_loss = 0
        if blur_lq_list != []:
            num_task+=1
            blur_un = torch.mean(torch.stack(blur_un_list),dim=0)
            s = 1.0/blur_un
            # blur_list_=torch.mul(blur_lq_list,blur_un)
            # blur_lq_list_ = torch.mul(blur_lq_list,blur_un)
            for u,v in zip(blur_list,blur_lq_list):
                blur_list_.append(torch.mul(u,s))
                blur_lq_list_.append(torch.mul(v,s))
            old_loss_dict[0]= (F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')).detach()   
            uncertainty_loss_dict[0] = (F.l1_loss(torch.stack(blur_lq_list_), torch.stack(blur_list_), reduction='mean')).detach()
            uncertainty_bn_dict[0] = 2*torch.mean(blur_un)
            blur_loss = F.l1_loss(torch.stack(blur_lq_list_), torch.stack(blur_list_), reduction='mean') + 2*torch.log(torch.mean(blur_un))
            #uncertainty项的值全为负数
            
        noise_loss = 0
        if noise_lq_list != []:
            num_task += 1
            noise_un = torch.mean(torch.stack(noise_un_list), dim=0)
            s = 1.0/noise_un #torch.exp(-noise_un)
            # noise_list_ = torch.mul(noise_lq_list, noise_un)
            # noise_lq_list_ = torch.mul(noise_lq_list, noise_un)
            for u, v in zip(noise_list, noise_lq_list):
                noise_list_.append(torch.mul(u, s))
                noise_lq_list_.append(torch.mul(v, s))
            old_loss_dict[1]= (F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')).detach()   
            uncertainty_loss_dict[1] = (F.l1_loss(torch.stack(noise_lq_list_), torch.stack(noise_list_), reduction='mean')).detach()
            uncertainty_bn_dict[1] = 2*torch.mean(noise_un)
            noise_loss = F.l1_loss(torch.stack(noise_lq_list_), torch.stack(noise_list_), reduction='mean') + 2*torch.log(torch.mean(noise_un))

        jpeg_loss = 0
        if jpeg_lq_list != []:
            num_task += 1
            jpeg_un = torch.mean(torch.stack(jpeg_un_list), dim=0)
            s = 1.0/jpeg_un
            for u, v in zip(jpeg_list, jpeg_lq_list):
                jpeg_list_.append(torch.mul(u, s))
                jpeg_lq_list_.append(torch.mul(v, s))
            # jpeg_list_ = torch.mul(jpeg_lq_list, jpeg_un)
            # jpeg_lq_list_ = torch.mul(jpeg_lq_list, jpeg_un)
            old_loss_dict[2]= (F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')).detach()   
            uncertainty_loss_dict[2] = (F.l1_loss(torch.stack(jpeg_lq_list_), torch.stack(jpeg_list_), reduction='mean')).detach()
            uncertainty_bn_dict[2] = 2*torch.mean(jpeg_un)
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list_), torch.stack(jpeg_list_), reduction='mean') + 2*torch.log(torch.mean(jpeg_un))
        
        
        rain_loss = 0
        if rain_lq_list != []:
            num_task += 1
            rain_un = torch.mean(torch.stack(rain_un_list), dim=0)
            s = 1.0/rain_un
            for u, v in zip(rain_list, rain_lq_list):
                rain_list_.append(torch.mul(u, s))
                rain_lq_list_.append(torch.mul(v, s))
            # rain_list_ = torch.mul(rain_lq_list, rain_un)
            # rain_lq_list_ = torch.mul(rain_lq_list, rain_un)
            old_loss_dict[3]= (F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')).detach()   
            uncertainty_loss_dict[3] = (F.l1_loss(torch.stack(rain_lq_list_), torch.stack(rain_list_), reduction='mean')).detach()
            uncertainty_bn_dict[3] = 2*torch.mean(rain_un)
            rain_loss = F.l1_loss(torch.stack(rain_lq_list_), torch.stack(rain_list_), reduction='mean')+ 2*torch.log(torch.mean(rain_un))

        haze_loss = 0
        if haze_lq_list != []:
            num_task += 1
            haze_un = torch.mean(torch.stack(haze_un_list), dim=0)
            s = 1.0/haze_un
            for u, v in zip(haze_list, haze_lq_list):
                haze_list_.append(torch.mul(u, s))
                haze_lq_list_.append(torch.mul(v, s))
            old_loss_dict[4]= (F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')).detach()   
            uncertainty_loss_dict[4] = (F.l1_loss(torch.stack(haze_lq_list_), torch.stack(haze_list_), reduction='mean')).detach()
            uncertainty_bn_dict[4] = 2*torch.mean(haze_un)
            # haze_list_ = torch.mul(haze_lq_list, haze_un)
            # haze_lq_list_ = torch.mul(haze_lq_list, haze_un)
            haze_loss = F.l1_loss(torch.stack(haze_lq_list_), torch.stack(haze_list_), reduction='mean')+ 2*torch.log(torch.mean(haze_un))

        dark_loss = 0
        if dark_lq_list != []:
            num_task += 1
            dark_un = torch.mean(torch.stack(dark_un_list), dim=0)
            s = 1.0/dark_un
            for u, v in zip(dark_list, dark_lq_list):
                dark_list_.append(torch.mul(u, s))
                dark_lq_list_.append(torch.mul(v, s))
            old_loss_dict[5]= (F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')).detach()   
            uncertainty_loss_dict[5] = (F.l1_loss(torch.stack(dark_lq_list_), torch.stack(dark_list_), reduction='mean')).detach()
            uncertainty_bn_dict[5] = 2*torch.mean(dark_un)
            # dark_list_ = torch.mul(dark_lq_list, dark_un)
            # dark_lq_list_ = torch.mul(dark_lq_list, dark_un)
            dark_loss = F.l1_loss(torch.stack(dark_lq_list_), torch.stack(dark_list_), reduction='mean')+ 2*torch.log(torch.mean(dark_un))

        sr_loss = 0
        if sr_lq_list != []:
            num_task += 1
            sr_un = torch.mean(torch.stack(sr_un_list), dim=0)
            s = 1.0/sr_un
            for u, v in zip(sr_list, sr_lq_list):
                sr_list_.append(torch.mul(u, s))
                sr_lq_list_.append(torch.mul(v, s))
            old_loss_dict[6]= (F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')).detach()   
            uncertainty_loss_dict[6] = (F.l1_loss(torch.stack(sr_lq_list_), torch.stack(sr_list_), reduction='mean')).detach()
            uncertainty_bn_dict[6] = 2*torch.mean(sr_un)
            # sr_list_ = torch.mul(sr_lq_list, sr_un)
            # sr_lq_list_ = torch.mul(sr_lq_list, sr_un)
            sr_loss = F.l1_loss(torch.stack(sr_lq_list_), torch.stack(sr_list_), reduction='mean')+ 2*torch.log(torch.mean(sr_un))
        
        total_loss = (blur_loss+noise_loss+jpeg_loss+rain_loss+haze_loss+dark_loss+sr_loss)/num_task
       
        return total_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss],old_loss_dict,uncertainty_loss_dict,uncertainty_bn_dict
    

