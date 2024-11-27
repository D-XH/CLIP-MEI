import torch
import torch.nn as nn
# from utils.utils import split_first_dim_linear
from utils.utils import split_first_dim_linear
import torchvision.models as models
import torch.nn.functional as F

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

#=========================================================================================================================#

def cls_d(x):
    # (5, 8, 2048)
    prototypes = x.mean(1)
    diff = prototypes.unsqueeze(1) - prototypes.unsqueeze(0)
    # 计算平方和
    square_diff = torch.sum(diff ** 2, dim=2)  # (5, 5)
    # 防止数值不稳定，确保没有负数或零
    square_diff = torch.clamp(square_diff, min=1e-12)
    distances = torch.sqrt(square_diff)
    # print(distances)
    mask = torch.ones_like(distances)
    torch.diagonal(mask)[:] = 0
    distances = distances * mask
    loss = -torch.mean(distances)
    return loss
import time

class sa_f(nn.Module):
    def __init__(self, seq_len=8, attn_drop=0., proj_drop=0., num_heads=8, dim=256):
        super().__init__()
        self.embed_dim = dim
        self.seq_len = seq_len
        self.pe = PatchEmbed(224, embed_dim=dim)
        self.pnum = self.pe.num_patches

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*2), act_layer=nn.GELU, drop=proj_drop)
        self.rech = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, su, qu):
        # su: (40, 3, 224, 224) qu: (8, 3, 224, 224)
        f_s = self.pe(su)
        f_q = self.pe(qu)
        f_s_t = f_s.reshape(-1, self.seq_len, self.pnum, self.embed_dim)
        f_q_t = f_q.unsqueeze(0)

        q = self.q(f_q_t) # (1, 8, 196, 768)
        kv = self.kv(f_s_t).reshape(-1, self.seq_len, self.pnum, 2, self.embed_dim).permute(3, 0, 1, 2, 4)
        k, v = kv[0], kv[1] # (5, 8, 196, 768)
        bq = q.size(0)
        bs = k.size(0)

        head_dim = self.embed_dim // self.num_heads
        q = q.reshape(bq, self.seq_len, self.pnum, self.num_heads, head_dim).permute(0, 1, 3, 2, 4) # (1, 8, 8, 196, 96)
        k = k.reshape(bs, self.seq_len, self.pnum, self.num_heads, head_dim).permute(0, 1, 3, 2, 4) # (5, 8, 8, 196, 96)
        v = v.reshape(bs, self.seq_len, self.pnum, self.num_heads, head_dim).permute(0, 1, 3, 2, 4) # (5, 8, 8, 196, 96)

        att = self.softmax((q * self.scale) @ k.transpose(-2, -1))  # (5, 8, 8, 196, 196)
        att = self.attn_drop(att)
        v = att @ v # (5, 8, 8, 196, 96)

        v = v.transpose(-3, -2).contiguous().reshape(su.size(0), self.pnum, self.embed_dim) # (40, 196, 768)
        v = self.proj(v)
        v = self.proj_drop(v)

        v = self.mlp(self.norm2(f_s + v)) + v

        v = self.rech(v.transpose(-2, -1).reshape(su.size(0), self.embed_dim, 14, 14))
        res = F.interpolate(v, size=(224, 224), mode='bilinear')
        return res

class mi_f(nn.Module):
    def __init__(self, seq_len=8, dim=256, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.embed_dim = dim
        self.seq_len = seq_len
        self.pe = PatchEmbed(224, embed_dim=dim)
        self.pnum = self.pe.num_patches
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2*dim)
        self.scale = dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*2), act_layer=nn.GELU, drop=proj_drop)

    def forward(self, x):
        # (40, 3, 224, 224)
        vid_num = x.size(0) // self.seq_len

        x = self.pe(x).reshape(vid_num, self.seq_len, self.pnum, self.embed_dim)  # (5, 8, 196, 256)

        q = self.q(x[:, 1:])
        q_last = self.q(x[:, -2]).unsqueeze(1)
        kv = self.kv(x[:, :-1]).reshape(vid_num, -1, self.pnum, 2, self.embed_dim).permute(3, 0, 1, 2, 4)
        kv_last = self.kv(x[:, -1]).reshape(vid_num, -1, self.pnum, 2, self.embed_dim).permute(3, 0, 1, 2, 4)
        k, v = kv[0], kv[1]
        k_last, v_last = kv_last[0], kv_last[1]

        att = self.softmax((q @ k.transpose(-2, -1)) * self.scale)  # (5, 7, 196, 196)
        att = self.attn_drop(att)

        att_l = self.softmax((q_last @ k_last.transpose(-2, -1)) * self.scale)  # (5, 1, 196, 196)
        att_l = self.attn_drop(att_l)

        res = att @ v   # (5, 7, 196, 256)
        res = torch.concat([res, att_l @ v_last], dim=1)   # (5, 8, 196, 256)

        res = self.proj(res)
        res = self.proj_drop(res)

        res = self.mlp(self.norm2(res))

        return res

class mix(nn.Module):
    def __init__(self, seq_len=8, embed_dim=256, trans_linear_in_dim=1024):
        super().__init__()
        self.seq_len = seq_len
        self.dim = trans_linear_in_dim
        self.pnum = 196
        self.sm_linear = nn.Linear(embed_dim, self.dim)
        self.qm_linear = nn.Linear(embed_dim, self.dim)
        self.q = nn.Linear(self.dim, self.dim)
        self.kv = nn.Linear(self.dim, self.dim*2)


    def forward(self, su, qu, s_m, q_m):
        # su:(40, 1024, 14, 14) qu:(160, 1024, 14, 14) s_m:(5, 8, 196, 256) q_m:(20, 8, 196, 256)
        
        query = qu.reshape(-1, self.seq_len, self.dim, self.pnum).transpose(-2, -1)   # (20, 8, 196, 1024)
        support = su.reshape(-1, self.seq_len, self.dim, self.pnum).transpose(-2, -1)   # (5, 8, 196, 1024)

        q = self.q(query)   # (20, 8, 196, 1024)
        kv = self.kv(support).reshape(-1, self.seq_len, self.pnum, 2, self.dim).permute(3, 0, 1, 2, 4) # (5, 8, 196, 1024*2)
        k, v = kv[0], kv[1]
        s_m = self.sm_linear(s_m)   # (5, 8, 196, 1024)
        q_m = self.qm_linear(q_m)   # (20, 8, 196, 1024)
        
        agent_attn = (s_m) @ k.transpose(-2, -1) # (5, 8, 196, 196)
        agent_v = agent_attn @ v    # (5, 8, 196, 1024)
        agent_v = agent_v.unsqueeze(0)

        q_attn = (q) @ q_m.transpose(-2, -1) # (20, 8, 196, 196)
        su_q = q_attn.unsqueeze(1) @ agent_v    # (20, 5, 8, 196, 1024)
        qu_q = q_attn @ q   #(20, 8, 196, 1024)

        su_q = su_q.mean(-2)    # (20, 5, 8, 1024)
        qu_q = qu_q.mean(-2)    # (20, 8, 1024)

        return su_q, qu_q

class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.train()
        self.cfg = cfg
        if self.cfg.MODEL.BACKBONE == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif self.cfg.MODEL.BACKBONE == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif self.cfg.MODEL.BACKBONE == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        last_layer_idx = -3
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])

        if self.cfg.MODEL.BACKBONE == "resnet50":
            trans_linear_in_dim = 1024
        else:
            trans_linear_in_dim = 256

        self.dim = trans_linear_in_dim
        self.seq_len = cfg.DATA.SEQ_LEN
        att_drop = cfg.MODEL.ATT_DROP
        proj_drop = cfg.MODEL.PROJ_DROP
        head_num = cfg.MODEL.HEAD_NUM
        embed_dim = cfg.MODEL.EMBED_DIM

        self.sa = sa_f(seq_len=self.seq_len, 
                       attn_drop=att_drop, 
                       proj_drop=proj_drop, 
                       num_heads=head_num, 
                       dim=embed_dim)
        self.motion_info = mi_f(seq_len=self.seq_len, 
                                dim=embed_dim, 
                                attn_drop=att_drop, 
                                proj_drop=proj_drop)
        self.mixer = mix(embed_dim=embed_dim, 
                         trans_linear_in_dim=self.dim)

    def forward(self, inputs):
        su, su_l, qu = inputs['context_images'], inputs['context_labels'], inputs['target_images']
        # su:(40, 3, 224, 224) qu:(160, 3, 224, 224)
        unique_labels = torch.unique(su_l)

        su_m = self.motion_info(su) # (5, 8, 196, 256)
        qu_m = self.motion_info(qu) # (20, 8, 196, 256)

        proto_q = qu.reshape(-1, self.seq_len, 3, 224, 224).mean(0)
        su = self.sa(su, proto_q) + su

        su_f = self.resnet(su)  # (40, 1024, 14, 14)
        qu_f = self.resnet(qu)  # (160, 1024, 14, 14)

        su_f, qu_f = self.mixer(su_f, qu_f, su_m, qu_m) # su:(20, 5, 8, 1024) qu:(20, 8, 1024)
        assert su_f.size(0) == qu_f.size(0)
        qn = qu_f.size(0)
        all_dists = []
        for i in range(qn):
            q = qu_f[i] # (8, 1024)
            s = su_f[i].reshape(-1, self.dim) # (40, 1024)
            frame_sim = cos_sim(q, s)    # [8, 40]
            frame_dists = 1 - frame_sim
            dists = frame_dists.reshape(1, 8, 5, 8).permute(0, 2, 1, 3)
            cum_dists = OTAM_cum_dist(dists) + OTAM_cum_dist(dists.transpose(-2, -1))

            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(su_l, c)), dim=1) for c in unique_labels]
            class_dists = torch.stack(class_dists).transpose(0, 1)  # (1, C)
            all_dists.append(class_dists)
        all_dists = torch.concat(all_dists, dim=0)
        # return_dict = {'logits': - all_dists}
        return_dict = {'logits': split_first_dim_linear(all_dists, [1, qn])}
        return return_dict
    
    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.cfg.DEVICE.NUM_GPUS > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.cfg.DEVICE.NUM_GPUS)])

            self.transformers.cuda(0)
            self.new_dist_loss_post_pat = [n.cuda(0) for n in self.new_dist_loss_post_pat]

            self.attn_pat.cuda(0)
            self.attn_pat = torch.nn.DataParallel(self.attn_pat, device_ids=[i for i in range(0, self.cfg.DEVICE.NUM_GPUS)])

            self.fr_enrich.cuda(0)
            self.fr_enrich = torch.nn.DataParallel(self.fr_enrich, device_ids=[i for i in range(0, self.cfg.DEVICE.NUM_GPUS)])