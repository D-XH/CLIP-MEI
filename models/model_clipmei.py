import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange

import os
import torch
from .clip_fsar import load,tokenize
from .myRes import cos_sim, OTAM_cum_dist_v2, extract_class_indices, Transformer_v1, FeedForward, Mlp, PositionalEncoder

class CNN(nn.Module):

    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.args = cfg
        if cfg.MODEL.BACKBONE == "RN50":
            clip_backbone, self.preprocess = load(cfg.MODEL.BACKBONE, device="cuda", cfg=cfg, jit=False)  # ViT-B/16
            self.backbone = clip_backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME  #所有得训练样本 标签名称
            self.class_real_test = cfg.TEST.CLASS_NAME    #所有得测试样本 标签名称
            self.mid_dim = 1024
        elif cfg.MODEL.BACKBONE == "ViT-B/16":
            #将clip模型加载了进来
            clip_backbone, self.preprocess = load(cfg.MODEL.BACKBONE, device="cuda", cfg=cfg, jit=False)  # ViT-B/16
            self.backbone = clip_backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            # backbone, self.preprocess = load("RN50", device="cuda", cfg=cfg, jit=False)
            # self.backbone = backbone.visual model.load_state_dict(state_dict)
            # self.backbone = CLIP
            self.mid_dim = 512
        with torch.no_grad():
            
            text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
            text_templete = tokenize(text_templete).cuda()
            self.text_features_train = clip_backbone.encode_text(text_templete)  # 对文本进行编码，数量就是support 中类别数量 例如 kinetics就是分别对64个单词进行编码

            text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
            text_templete = tokenize(text_templete).cuda()
            self.text_features_test = clip_backbone.encode_text(text_templete)

        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)
        
        # set_trace()
        self.factor=1
        self.motion_conv1 = nn.Conv1d(self.mid_dim // self.factor, self.mid_dim // self.factor, kernel_size=3, padding=1,groups=1)
        self.motion_conv2 = nn.Conv1d(self.mid_dim // self.factor, self.mid_dim // self.factor, kernel_size=3, padding=1, groups=1)

        self.token_tr = token_trans(self.mid_dim)
        self.context1 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=256, dropout_atte=0.2)
        self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=256, dropout_atte=0.2)
        self.mo_alpha1 = nn.Parameter(torch.rand(1), requires_grad=True)
        nn.init.constant_(self.mo_alpha1, 1)
        
    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CLIP visual encoder features.
        """
        if self.training:
            support_features = self.backbone(support_images).squeeze()  # self.backbone 为 clip的视觉分支visual ModifeidResnet 输出维度 [(way*shot*frames_number),1024]
            # os.system("nvidia-smi")
            target_features = self.backbone(target_images).squeeze()
            # os.system("nvidia-smi")
            dim = int(support_features.shape[1])
            support_features = support_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            target_features = target_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            support_features_text = None
        else:
            support_features = self.backbone(support_images).squeeze()
            target_features = self.backbone(target_images).squeeze()
            dim = int(target_features.shape[1])

            support_features = support_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            target_features = target_features.reshape(-1, self.args.DATA.SEQ_LEN, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            # support_real_class = torch.unique(support_real_class)
            support_features_text = self.text_features_test[support_real_class.long()]  # self.text_features_test 为clip的 文本分支 backbone.encode_text

        return support_features, target_features, support_features_text

    def get_motion_feats(self, support_features, target_features):

        support_features_conv = self.motion_conv1(support_features.permute(0,2,1))
        target_features_conv = self.motion_conv1(target_features.permute(0, 2, 1))
        support_features_conv = self.motion_conv2(support_features_conv)
        target_features_conv = self.motion_conv2(target_features_conv)

        support_features_motion_f = support_features_conv[:,:, 1:] - support_features.permute(0,2,1)[:,:, :-1]
        target_features_motion_f = target_features_conv[:,:, 1:] - target_features.permute(0,2,1)[:,:, :-1]

        support_features_motion_b = support_features_conv[:,:, :-1] - support_features.permute(0,2,1)[:,:, 1:]
        target_features_motion_b = target_features_conv[:,:, :-1] - target_features.permute(0,2,1)[:,:, 1:]

        support_features_motion = 0.5*(support_features_motion_f + support_features_motion_b)
        target_features_motion = 0.5*(target_features_motion_f + target_features_motion_b)

        return support_features_motion.mean(-1), target_features_motion.mean(-1)
    
    def forward(self, inputs):  # 获得support support labels, query, support real class
        support_images, support_labels, target_images, support_real_class = inputs['context_images'], inputs['context_labels'], inputs['target_images'], inputs['real_support_labels']
        target_real_class = inputs['real_target_labels']
        # set_trace()
        #一个episode中所有的样本对应的文本语句特征
        if self.training:  # 取5个support样本类型对应的样本标签编码，输出维度为5 1 1024
            self.context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.SEQ_LEN, 1)
            self.target_context_support = self.text_features_train[target_real_class.long()].unsqueeze(1)
        else:
            self.context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.SEQ_LEN, 1) # .repeat(support_bs+target_bs, 1, 1)
            self.target_context_support = self.text_features_test[target_real_class.long()].unsqueeze(1)

        support_features, target_features, text_features = self.get_feats(support_images, target_images, support_real_class)

        su, qu = support_features, target_features 
        mo_dist_pre = self.mo(su, qu)    # MIC

        su, qu, su_t2, qu_t2, class_dists_l, consist_distance, text_distance = self.cpt_sem(su, qu, support_labels)    # QSA
        dists = consist_distance + text_distance + self.mo_alpha1*mo_dist_pre

        su_2, qu_2, su_t2, qu_t2 = self.taskM(su, qu, support_labels)    # TFE
        unique_labels = torch.unique(support_labels)
        su_pro = [
           torch.mean(torch.index_select(su_2, 0, extract_class_indices(support_labels, c)), dim=0)
           for c in unique_labels]
        su_pro = torch.stack(su_pro)
        task_dist = self.otam_distance(su_pro, qu_2) + self.otam_distance(su_t2, qu_t2)

        return_dict = {
                       "logits": - (0.5*class_dists_l + task_dist).unsqueeze(0),    #帧匹配
                       "dists": dists,
                       }  # [5， 5] , [10 64]
        return return_dict
    
    def mo(self, su, qu):
        # (25, 8, 1024) (20, 8, 1024)
        su_mo, qu_mo = self.get_motion_feats(su, qu)  # (25, 1024) (20, 1024)
        su_m = su + 0.1 * su_mo.unsqueeze(1)
        qu_m = qu + 0.1 * qu_mo.unsqueeze(1)
        su_m = torch.concat([su_mo.unsqueeze(1), su_m], dim=1)   # (25, 9, 1024)
        qu_m = torch.concat([qu_mo.unsqueeze(1), qu_m], dim=1)   # (20, 9, 1024)

        qu_m = self.context2(qu_m, qu_m, qu_m)
        su_m = self.context2(su_m, su_m, su_m)
        
        qu_mo = qu_m[:, 0, :]
        su_mo = su_m[:, 0, :]
        qu_m = qu_m[:, 1:, :]
        su_m = su_m[:, 1:, :]

        new_sm, new_qm = self.get_motion_feats(su_m, qu_m)  # (25, 1024) (20, 1024)

        qu_mo_dis = self._dis(new_qm, qu_mo)
        su_mo_dis = self._dis(new_sm, su_mo)

        dis = 1 * (qu_mo_dis + su_mo_dis)
        return dis
        
    def sem(self, su, qu, su_l):
        # (25, 8, 1024) (20, 8, 1024) (25,)
        ######################################################
        token = self.target_context_support.mean(0) # (1, 1, 1024)
        support_token = self.token_tr(token, su)   # (25, 1, 1024)
        target_token = self.token_tr(token, qu)   # (20, 1, 1024
        ######################################################

        # real query token, 只有训练的时候用
        qu_real, token_q_real = self.se_te(qu, self.target_context_support)
        # fake support token
        su_fake = 0

        # fake query token
        qu_fake, token_q_fake = self.se_te(qu, target_token)

        # real support token
        su_real, token_s_real = self.se_te(su, self.context_support)

        unique_labels = torch.unique(su_l)
        su_pro = [
            torch.mean(torch.index_select(su_real, 0, extract_class_indices(su_l, c)), dim=0)
            for c in unique_labels]
        # token_s_real = [
        #     torch.mean(torch.index_select(token_s_real, 0, extract_class_indices(su_l, c)), dim=0)
        #     for c in unique_labels]
        # token_s_real = torch.stack(token_s_real)
        su_pro = torch.stack(su_pro)
        return su_real, qu_fake, su_pro, su_fake, qu_real, support_token, target_token,  token_q_real, token_q_fake, token_s_real
    
    def cpt_sem(self, su, qu, su_l):
        su_real, qu_fake, su_pro, su_fake, qu_real, su_token, qu_token,  token_q_real, token_q_fake, token_s_real = self.sem(su, qu, su_l)
        # initial token --> qu
        token_dis = self._dis(self.target_context_support, qu_token)
        # initial token --> su
        # token_dis_1 = self._dis(self.context_support, su_token)
        # lastest token --> qu
        token_dis_2 = self._dis(token_q_real, token_q_fake)
        #token_t = token_q_real - token_q_fake   # 
        #token_t = torch.norm(token_t.squeeze(1), dim=[-2, -1]) ** 2
        #token_dis_2 = torch.mean(token_t)
        # lastest token --> su
        # token_dis_3 = self._dis(token_s_real, token_s_fake)
        # lastest token --> qu-su
        # token_dis_4 = self._dis(token_s_real, token_q_fake)

        text_distance = (token_dis + token_dis_2 ) * 0.5

        consist_distance = self._dis(torch.concat([token_q_real, qu_real], dim=1), torch.concat([token_q_fake, qu_fake],dim=1))

        cum_dists = self.otam_distance(su_pro, qu_fake)
        #t_dis = self.otam_distance(token_s_real, token_q_fake)
        class_dists_l = cum_dists #+ 0.8 * t_dis

        return su_real, qu_fake, token_s_real, token_q_fake, class_dists_l, consist_distance, text_distance
    
    def taskM(self, su, qu, su_l):
         # (25, 8, 256) (20, 8, 256)
         unique_labels = torch.unique(su_l)
         suu = [
             torch.index_select(su, 0, extract_class_indices(su_l, c))
             for c in unique_labels]
         suu = torch.stack(suu)  # (5, 5, 8, 1024)
         cn = suu.size(0)
         token_s = torch.mean(torch.concat([suu, qu.unsqueeze(0).repeat(cn,1,1,1)], dim=1), dim=1)    # (5, 8, 1024)
         token_q = torch.mean(token_s, dim=0, keepdim=True)  # (1, 8, 1024)
       
         su_t = torch.concat([token_s, su], dim=0).permute(1, 0, 2)  # (8, 30, 1024) # del
         qu_t = torch.concat([token_q, qu], dim=0).permute(1, 0, 2)  # (8, 21, 1024)

         _su = self.context1(su_t, su_t, su_t).permute(1, 0, 2)
         _qu = self.context1(qu_t, qu_t, qu_t).permute(1, 0, 2)

         return _su[cn:, :, :], _qu[1:, :, :], _su[:cn, :, :], _qu[0, :, :].unsqueeze(0)    

    def se_te(self, qu, token_q):
        # (20, 8, 1024) (25, 1, 1024) 
        q = qu + 0.1*token_q
        #token_q = self.trans2(token_q, qu, qu)
        q = torch.concat([token_q, q], dim=1)
        #kv = torch.concat([token_q, qu+0.1*token_q], dim=1)
        q = self.context2(q, q, q)  # (20, 9, 1024)
        token_q, q = q[:,0,:], q[:,1:,:]

        return q, token_q.unsqueeze(1)

    def video2imagetext_adapter_mean(self, support_features, target_features):
        if self.training:
            text_features = self.text_features_train
        else:
            text_features = self.text_features_test
        # 是否使用分类标签 也就是文本分支
        if hasattr(self.args.MODEL, "USE_CLASSIFICATION") and self.args.MODEL.USE_CLASSIFICATION:
            feature_classification_in = torch.cat([support_features, target_features],  dim=0)  # 第一维度是support query一共的样本数量
            feature_classification = feature_classification_in.mean(1)  # 在第二位 帧的维度进行平均  维度为 [样本数量，1024] 公>式2中的GAP操作
            class_text_logits = cos_sim(feature_classification, text_features) * self.scale  # 10 64， 10是10个视频样本，64是64个标签，对应于公式2
        else:
            class_text_logits = None
        return class_text_logits

    def _dis(self, x, y):
        diff = x - y  # (20, 1, 1024)
        dim = [-2, -1] if len(x.shape) == 3 else [-1]
        norm_sq = torch.norm(diff, dim=dim) ** 2
        distance = torch.mean(norm_sq)
        return distance
    
    def otam_distance(self, support_features, target_features):
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]
        support_features = rearrange(support_features, 'b s d -> (b s) d')  # 5 8 1024-->40  1024
        target_features = rearrange(target_features, 'b s d -> (b s) d')
        frame_sim = cos_sim(target_features, support_features)  # 类别数量*每个类的样本数量， 类别数量
        frame_dists = 1 - frame_sim
        # dists维度为 query样本数量， support类别数量，帧数，帧数
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 25, 8, 8]
        # calculate query -> support and support -> query  双向匹配还是单向匹配
        if hasattr(self.args.MODEL, "SINGLE_DIRECT") and self.args.MODEL.SINGLE_DIRECT:
            cum_dists = OTAM_cum_dist_v2(dists)
        else:
            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        return cum_dists
    
class token_trans(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.trans = Transformer_v1(dim=1024, heads=8, dim_head_k=256, dropout_atte=0.2, depth=1)
        self.mlp = FeedForward(dim, 2048, dropout=0.05)
        #self.mlp2 = FeedForward(1024, 2048, dropout=0.05)

    def forward(self, t, qu):
        # (1, 1, 1024) (20, 8, 1024)
        t = t.expand(qu.size(0), -1, -1)    # (20, 1, 1024)
        #x = self.trans(t, qu, qu)
        #x = self.mlp(x)
        x = self.mlp(t*qu.mean(dim=[1,2], keepdim=True))
        #x=self.mlp2(x)
        return x
    

