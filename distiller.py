import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np
import math


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
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

class SAST(nn.Module):
   
   def __init__(self, t_channel, s_channel):
      super(SAST, self).__init__()

      
      self.B = nn.Conv2d(s_channel, s_channel, kernel_size = 3, padding = 1)
      self.C = nn.Conv2d(s_channel, s_channel, kernel_size = 3, padding = 1)
    #   self.D = nn.Conv2d(s_channel, s_channel, kernel_size = 3, padding = 1)
    #   self.connector = nn.Conv2d(s_channel ,t_channel, kernel_size = 1)

      self.alpha = 1


   def forward(self, x):
      b, c, h, w = x.shape
      M = h * w
      A_b = self.B(x).reshape(b, M, c)
      A_c = self.C(x).reshape(b, M, c)
    #   A_d = self.D(x).reshape(b, M, c)

      # b x M x c * b x c x M = b x M x M
      S = torch.bmm(A_b, A_c.permute(0,2,1)) / np.sqrt(M)

      identity_mask = torch.eye(M, dtype=S.dtype, device=S.device).unsqueeze(0).expand(b, -1, -1)

      S = S * (1 - identity_mask)


      S = torch.softmax(S, dim = 2)
        
      return S
   

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

        self.SAST = SAST(t_channels[3], s_channels[3])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net
        self.args = args
        self.loss_divider = [8, 4, 2, 1, 1, 4*4]
        self.criterion = sim_dis_compute
        self.temperature = 1
        self.scale = 0.5

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)

        pa_loss = 0 
        if self.args.pa_lambda is not None: # pairwise loss
          feat_T = t_feats[4]
          feat_S = s_feats[4]
          total_w, total_h = feat_T.shape[2], feat_T.shape[3]
          patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
          maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
          pa_loss = self.args.pa_lambda * self.criterion(maxpool(feat_S), maxpool(feat_T))
   
        # Wrong?
        pi_loss = 0
        if self.args.pi_lambda is not None: # pixelwise loss
          #TF = F.normalize(t_feats[5].pow(2).mean(1)) 
          #SF = F.normalize(s_feats[5].pow(2).mean(1)) 
          #pi_loss = self.args.pi_lambda * (TF - SF).pow(2).mean()
          pi_loss =  self.args.pi_lambda * torch.nn.KLDivLoss()(F.log_softmax(s_out / self.temperature, dim=1), F.softmax(t_out / self.temperature, dim=1))


        SA_loss = 0
        if self.args.SA_lambda is not None: # Selt-attention loss
           
           layer = 3

           b,c,h,w = t_feats[layer].shape

           TF = t_feats[layer] # b x c' x h x w
           SF = s_feats[layer] # b x c x h x w

           # h and w are the same
           
           M = h * w

           TF = TF.view(b,M,c)

           # b x M x M
           X = torch.bmm(TF, TF.permute(0,2,1)) / np.sqrt(M)

           identity_mask = torch.eye(M, dtype=X.dtype, device=X.device).unsqueeze(0).expand(b, -1, -1)
           
           X = X * (1 - identity_mask)

           X = F.log_softmax(X, dim = 2) 
           
           # b x M x M
           S = self.SAST(SF)

           SA_loss = (X - S) ** 2
           SA_loss = SA_loss.sum() ** 0.5
           SA_loss = SA_loss / (b * M * M)
  
           SA_loss = self.args.SA_lambda * SA_loss



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
        
        
        
        lo_loss = 0
        if self.args.lo_lambda is not None: 
          b, c, h, w = s_out.shape
          s_logit = torch.reshape(s_out, (b, c, h*w))
          t_logit = torch.reshape(t_out, (b, c, h*w))

          s_logit = F.softmax(s_out / self.temperature, dim=2)
          t_logit = F.softmax(t_out / self.temperature, dim=2)
          kl = torch.nn.KLDivLoss(reduction="batchmean")
          ICCS = torch.empty((21,21)).cuda()
          ICCT = torch.empty((21,21)).cuda()
          for i in range(21):
            for j in range(i, 21):
              ICCS[j, i] = ICCS[i, j] = kl(s_logit[:, i], s_logit[:, j])
              ICCT[j, i] = ICCT[i, j] = kl(t_logit[:, i], t_logit[:, j])

          ICCS = torch.nn.functional.normalize(ICCS, dim = 1)
          ICCT = torch.nn.functional.normalize(ICCT, dim = 1)
          lo_loss =  self.args.lo_lambda * (ICCS - ICCT).pow(2).mean()/b 

        return s_out, pa_loss, pi_loss, ic_loss, lo_loss, SA_loss, AG_loss
