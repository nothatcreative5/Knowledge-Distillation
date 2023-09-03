import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np
import math
from utils.loss import SegmentationLosses


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

# Makes sense
def dist_loss(source, target):
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=3, stride=1, padding=1, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


   
def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff

   
def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

    

class Distiller(nn.Module):
    def __init__(self, t_net, s_net, args):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        encoder_layer = nn.TransformerEncoderLayer(d_model=s_channels[3], nhead=8, batch_first = True, dropout = 0.5)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net
        self.args = args
        self.optimizer = torch.optim.SGD(self.t_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.loss_divider = [8, 4, 2, 1, 1, 4*4]
        # self.criterion = sim_dis_compute
        self.temperature = 1
        self.scale = 0.5

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)



        SA_loss = 0
        if self.args.sa_lambda is not None:
            layer = 3
            TF = t_feats[layer]
            SF = self.Connectors[layer](s_feats[layer])

            b,c,h,w = TF.shape

            M = h*w

            # (b, c, h, w) -> (b, c)
            TF_c = torch.sum(torch.abs(TF), dim = [2,3])
            TF = TF.reshape(b,c,-1)
            # (B,C, HW, HW)
            ATT_T = torch.einsum('bij, bki -> bijk', TF,torch.permute(TF, (0, 2, 1)))
            ATT_T = torch.einsum('bijk, bi -> bjk', ATT_T, TF_c)
            ATT_T = torch.nn.functional.normalize(ATT_T, dim = 2)


            SF_c = torch.sum(torch.abs(SF), dim = [2,3])
            SF = SF.reshape(b,c,-1)
            ATT_S = torch.einsum('bij, bki -> bijk', SF,torch.permute(SF, (0, 2, 1)))
            ATT_S = torch.einsum('bijk, bi -> bjk', ATT_S, SF_c)
            ATT_S = torch.nn.functional.normalize(ATT_S, dim = 2)

            Att_diff = ATT_S - ATT_T
            SA_loss = self.args.sa_lambda * (Att_diff * Att_diff).view(b, -1).sum() / (b * M ** 2)



        # Correct
        ic_loss = 0
        if self.args.ic_lambda is not None: #logits loss
            b, c, h, w = s_out.shape

            s_logit = torch.reshape(s_out, (b, c, h*w))
            t_logit = torch.reshape(t_out, (b, c, h*w))

            # b x c x A  mul  b x A x c -> b x c x c
            ICCT = torch.bmm(t_logit, t_logit.permute(0,2,1))
            ICCT = torch.nn.functional.normalize(ICCT, dim = 2)

            ICCS = torch.bmm(s_logit, s_logit.permute(0,2,1))
            ICCS = torch.nn.functional.normalize(ICCS, dim = 2)

            G_diff = ICCS - ICCT
            ic_loss = self.args.ic_lambda * (G_diff * G_diff).view(b, -1).sum() / (c*b)

        return s_out,ic_loss, SA_loss
