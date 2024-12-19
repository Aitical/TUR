from torch import nn as nn
from torch.nn import functional as F
import torch

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer
from basicsr.archs.airnet.encoder import CBDE
from basicsr.archs.airnet.DGRN import DGRN,DGRN_uncertainty

@ARCH_REGISTRY.register()
class AirNet_uncertainty(nn.Module):
    def __init__(self, fix_E=True,if_uncertainty=True,batch_size=32):
        super(AirNet_uncertainty, self).__init__()
        self.if_uncertainty = if_uncertainty
        self.batch_size = batch_size
        # Restorer 
        if self.if_uncertainty:
            self.R = DGRN_uncertainty()
        else:
            self.R = DGRN()
            
        self.fix_E = fix_E
        # Encoder  
        self.E = CBDE(self.batch_size)
        self.freeze_encoder_parameters()
        
        
    def freeze_encoder_parameters(self):
        for param in self.E.parameters():
            param.requires_grad = False

    def forward(self, x_query, x_key=None):
        if self.training:
            if self.fix_E:
                fea, inter = self.E.inference(x_query)
            else:
                fea, logits, labels, inter = self.E(x_query, x_key)
            if self.if_uncertainty:
                restored,un = self.R(x_query, inter)
                return restored,un
            else:
                restored = self.R(x_query,inter)
                return restored
        else:
            fea, inter = self.E(x_query, x_query)
            
            if self.if_uncertainty:
                restored,un = self.R(x_query, inter)
                return restored,un
            else:
                restored = self.R(x_query,inter)
                return restored
            


@ARCH_REGISTRY.register()
class AirNet_uncertainty_with_encoder(nn.Module):
    def __init__(self, fix_E=True,if_uncertainty=True,batch_size=32):
        super(AirNet_uncertainty_with_encoder, self).__init__()
        self.if_uncertainty = if_uncertainty
        self.batch_size = batch_size
        # Restorer 
        if self.if_uncertainty:
            self.R = DGRN_uncertainty()
        else:
            self.R = DGRN()
            
        self.fix_E = fix_E
        # Encoder  
        self.E = CBDE(self.batch_size)
        #self.freeze_encoder_parameters()
        
        
    def freeze_encoder_parameters(self):
        for param in self.E.parameters():
            param.requires_grad = False

    def forward(self, x_query, x_key=None):
        if self.training:
            if self.fix_E:
                fea, inter = self.E.inference(x_query)
            else:
                fea, logits, labels, inter = self.E(x_query, x_key)
            if self.if_uncertainty:
                restored,un = self.R(x_query, inter)
                return restored,un
            else:
                restored = self.R(x_query,inter)
                return restored
        else:
            fea, inter = self.E(x_query, x_query)
            
            if self.if_uncertainty:
                restored,un = self.R(x_query, inter)
                return restored,un
            else:
                restored = self.R(x_query,inter)
                return restored