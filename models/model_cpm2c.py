import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange

import os
import torch
from .clip_fsar import load,tokenize
from .myRes import cos_sim, OTAM_cum_dist_v2, extract_class_indices, Transformer_v1, FeedForward


class CLIP_CPMMC_FSAR(nn.Module):

    def __init__(self, cfg):
        super(CLIP_CPMMC_FSAR, self).__init__()
        self.args = cfg
        if cfg.MODEL.BACKBONE == "RN50":
            clip_backbone, self.preprocess = load(cfg.MODEL.BACKBONE, device="cuda", cfg=cfg, jit=False)  # ViT-B/16
            self.backbone = clip_backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME  #所有得训练样本 标签名称
            self.class_real_test = cfg.TEST.CLASS_NAME    #所有得测试样本 标签名称
            self.mid_dim = 1024
            
            #for name, param in self.backbone.named_parameters():
            ##    if 'layer3' in name:
            #        break
            #    param.requires_grad = False
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

        self.mid_layer = nn.Sequential()
        self.classification_layer = nn.Sequential()
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)

        if hasattr(self.args.MODEL, "TRANSFORMER_DEPTH") and self.args.MODEL.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.args.MODEL.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)
        # set_trace()
        self.factor=1
        self.motion_conv1 = nn.Conv1d(self.mid_dim // self.factor, self.mid_dim // self.factor, kernel_size=3, padding=1,groups=1)
        self.motion_conv2 = nn.Conv1d(self.mid_dim // self.factor, self.mid_dim // self.factor, kernel_size=3, padding=1, groups=1)
        self.motion_coeff = self.args.MODEL.MOTION_COFF
        self.normal_coeff = self.args.MODEL.NORMAL_COFF
        #self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        #self.class_token_motion = nn.Parameter(torch.randn(1, 1, self.mid_dim))

        #self.token_linear1 = nn.Linear(self.mid_dim // self.factor, self.mid_dim // self.factor)
        #self.token_linear2 = nn.Linear(self.mid_dim // self.factor, self.mid_dim // self.factor)
        self.token_tr = token_trans()
        #self.motion_tr = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)
        #self.momo = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)

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

        support_features_motion_f_r = support_features_conv[:,:, 1:] - support_features.permute(0,2,1)[:,:, :-1]
        target_features_motion_f_r = target_features_conv[:,:, 1:] - target_features.permute(0,2,1)[:,:, :-1]
        #support_features_motion_f = self.motion_tr(support_features_conv.permute(0,2,1)[:,1:,:], support_features[:,:-1,:], support_features[:,:-1,:])
        #target_features_motion_f = self.motion_tr(target_features_conv.permute(0,2,1)[:,1:,:], target_features[:,:-1,:], target_features[:,:-1,:])

        support_features_motion_b_r = support_features_conv[:,:, :-1] - support_features.permute(0,2,1)[:,:, 1:]
        target_features_motion_b_r = target_features_conv[:,:, :-1] - target_features.permute(0,2,1)[:,:, 1:]
        #support_features_motion_b = self.motion_tr(support_features_conv.permute(0,2,1)[:,:-1,:], support_features[:,1:,:], support_features[:,1:,:])
        #target_features_motion_b = self.motion_tr(target_features_conv.permute(0,2,1)[:,:-1,:], target_features[:,1:,:], target_features[:,1:,:])

        support_features_motion = 1*((0.5*support_features_motion_f_r.permute(0,2,1)) + (0.5*support_features_motion_b_r.permute(0,2,1)))
        target_features_motion = 1*((0.5*target_features_motion_f_r.permute(0,2,1)) + (0.5*target_features_motion_b_r.permute(0,2,1)))

        return support_features_motion, target_features_motion#.permute(0,2,1)

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
        support_features_motion, target_features_motion = self.get_motion_feats(support_features, target_features)
        class_text_logits = self.video2imagetext_adapter_mean(support_features, target_features)

        #mo_dis, support_features, target_features = self.mo(support_features, target_features)

        class_dists_g, class_dists_l, consist_distance , text_distance= self.train_and_test(self.context_support, support_features, support_features_motion, support_labels, target_features, target_features_motion)
        
        mo_dis = self.mo(support_features, target_features)
        #class_dists_g = rearrange(class_dists_g, 'c q -> q c')

        return_dict = {"class_logits": class_text_logits.unsqueeze(0),  #图文匹配
                       "logits_local": - class_dists_l.unsqueeze(0),    #帧匹配
                       "logits_global": -class_dists_g.unsqueeze(0),    #全局token匹配
                       "target_consist_distance": consist_distance,  #一致性
                       "text_distance": text_distance+mo_dis
                       }  # [5， 5] , [10 64]
        return return_dict

    def mo(self, su, qu):
        # (25, 8, 1024) (20, 8, 1024)
        su_mo, qu_mo = self.get_motion_feats(su, qu)  # (25, 1024) (20, 1024)
        su_mo = su_mo.mean(1, keepdim=True)
        qu_mo = qu_mo.mean(1, keepdim=True)
        su_m = su + 0.1 * su_mo
        qu_m = qu + 0.1 * qu_mo
        su_m = torch.concat([su_mo, su_m], dim=1)   # (25, 9, 1024)
        qu_m = torch.concat([qu_mo, qu_m], dim=1)   # (20, 9, 1024)

        qu_m = self.context2(qu_m, qu_m, qu_m)
        su_m = self.context2(su_m, su_m, su_m)

        qu_mo = qu_m[:, 0, :]
        su_mo = su_m[:, 0, :]
        qu_m = qu_m[:, 1:, :]
        su_m = su_m[:, 1:, :]

        new_sm, new_qm = self.get_motion_feats(su_m, qu_m)  # (25, 1024) (20, 1024)

        qu_mo_dis = self._dis(new_qm.mean(1), qu_mo)
        su_mo_dis = self._dis(new_sm.mean(1), su_mo)

        dis = 1 * (qu_mo_dis + su_mo_dis)
        return dis#, su_m+su, qu_m+qu
    def _dis(self, x, y):
        diff = x - y  # (20, 1, 1024)
        dim = [-2, -1] if len(x.shape) == 3 else [-1]
        norm_sq = torch.norm(diff, dim=dim) ** 2
        distance = torch.mean(norm_sq)
        return distance

    def train_and_test(self, context_support, support_features, support_features_motion, support_labels, target_features, target_features_motion):

        support_features_motion, target_features_motion, support_features_motion_pro, support_features_contra_motion, target_features_contra_motion, mo_t_token, mo_s_token = self.text_eh_temporal_transformer(context_support, support_features_motion, target_features_motion, support_labels)
        support_features, target_features, support_features_pro, support_features_contra, target_features_contra, t_token, s_token = self.text_eh_temporal_transformer(context_support, support_features, target_features, support_labels)

        ###############################
        token_diff = self.target_context_support - t_token  # (20, 1, 1024)
        token_norm_sq = torch.norm(token_diff, dim=[-2, -1]) ** 2
        token_distance = torch.mean(token_norm_sq)

        token_mo_diff = self.target_context_support - mo_t_token  
        token_mo_norm_sq = torch.norm(token_mo_diff, dim=[-2, -1]) ** 2
        token_mo_distance = torch.mean(token_mo_norm_sq)

        #token_1_diff = context_support - s_token  # (20, 1, 1024)
        #token_1_norm_sq = torch.norm(token_1_diff, dim=[-2, -1]) ** 2
        #token_1_distance = torch.mean(token_1_norm_sq)

        #token_mo_1_diff = context_support - mo_s_token
        #token_mo_1_norm_sq = torch.norm(token_mo_1_diff, dim=[-2, -1]) ** 2
        #token_mo_1_distance = torch.mean(token_mo_1_norm_sq)

        #token_2_diff = support_features_contra[:, 0, :] - support_features[:, 0, :]
        #token_2_norm_sq = torch.norm(token_2_diff, dim=[-2, -1]) ** 2
        #token_2_distance = torch.mean(token_2_norm_sq)

        #token_mo_2_diff = support_features_contra_motion[:, 0, :] - support_features_motion[:, 0, :]
        #token_mo_2_norm_sq = torch.norm(token_mo_2_diff, dim=[-2, -1]) ** 2
        #token_mo_2_distance = torch.mean(token_mo_2_norm_sq)

        token_3_diff = target_features_contra[:, 0, :] - target_features[:, 0, :]   # 
        token_3_norm_sq = torch.norm(token_3_diff, dim=[-2, -1]) ** 2
        token_3_distance = torch.mean(token_3_norm_sq)

        token_mo_3_diff = target_features_contra_motion[:, 0, :] - target_features_motion[:, 0, :]
        token_mo_3_norm_sq = torch.norm(token_mo_3_diff, dim=[-2, -1]) ** 2
        token_mo_3_distance = torch.mean(token_mo_3_norm_sq)

        text_distance = self.normal_coeff * (token_distance + token_3_distance) * 0.5 \
                        + self.motion_coeff * (token_mo_distance + token_mo_3_distance) * 0.5
        ###############################
        #dis_g = g_dis_c(support_features_pro[:, 0, :], support_labels, target_features[:, 0, :])
        #dis_g_mo = g_dis_c(support_features_motion_pro[:, 0, :], support_labels, target_features_motion[:, 0, :])
        #class_dists_g = self.normal_coeff * dis_g + self.motion_coeff * dis_g_mo
        #-----------------------------#

        #part 1, 在测试中这个距离是不用的
        ################################motion 一致性损失 start#######################################################
        target_diff_motion = target_features_motion - target_features_contra_motion
        target_norm_sq_motion = torch.norm(target_diff_motion, dim=[-2, -1]) ** 2
        # target_consist_distance_motion = torch.div(target_norm_sq_motion, target_features_motion.shape[0])
        target_consist_distance_motion = torch.mean(target_norm_sq_motion)
        #support_diff_motion = support_features_motion - support_features_contra_motion
        #support_norm_sq_motion = torch.norm(support_diff_motion, dim=[-2, -1]) ** 2
        # support_consist_distance_motion = torch.div(support_norm_sq_motion, target_features_motion.shape[0])
        #support_consist_distance_motion = torch.mean(support_norm_sq_motion)
        consist_distance_motion = target_consist_distance_motion
        ################################motion 一致性损失 end#######################################################

        ################################normal 一致性损失 start#######################################################
        target_diff = target_features - target_features_contra
        target_norm_sq = torch.norm(target_diff, dim=[-2, -1]) ** 2
        # target_consist_distance = torch.div(target_norm_sq, target_features.shape[0])
        target_consist_distance = torch.mean(target_norm_sq)
        #support_diff = support_features - support_features_contra
        #support_norm_sq = torch.norm(support_diff, dim=[-2, -1]) ** 2
        # support_consist_distance = torch.div(support_norm_sq, support_features.shape[0])
        #support_consist_distance = torch.mean(support_norm_sq)
        consist_distance = target_consist_distance
        #print("support_consist_distance " + str(support_consist_distance))
        #print("target_consist_distance " + str(target_consist_distance))
        ################################normal 一致性损失 end#######################################################
        #motion和normal总体一致性损失
        consist_distance = self.normal_coeff * consist_distance + self.motion_coeff * consist_distance_motion

        # part 2, 文章中part2 的系数为零，也就这部分代码其实不生效，仅我们实验中参考一下
        ################################motion 和 normal global 距离 start#########################################
        #support_features_motion_g = support_features_motion[:, 0, :]
        #class_dists_q2s_motion = self.global_distance(support_features_motion_g, support_labels, target_features_motion)

        #support_features_g = support_features[:, 0, :]
        #class_dists_q2s = self.global_distance(support_features_g, support_labels, target_features)

        #class_dists_g = self.normal_coeff * class_dists_q2s + self.motion_coeff * class_dists_q2s_motion
        ################################motion 和 normal global 距离 end###########################################

        # part 3, 可以按需要需要换下面的这个self.otam_distance的方法就可以了
        ################################motion和normal的局部帧对齐距离，以及其总体距离 start#########################################
        cum_dists = self.otam_distance(support_features_pro[:, 1:, :], target_features[:, 1:, :])
        cum_dists_motion = self.otam_distance(support_features_motion_pro[:, 1:, :], target_features_motion[:, 1:, :])


        class_dists_l = self.normal_coeff * cum_dists + self.motion_coeff * cum_dists_motion
        ################################motion和normal的局部帧对齐距离，以及其总体距离 end#########################################

        return class_dists_l, class_dists_l, consist_distance, text_distance
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
    def global_distance(self, support_features_g, support_labels, target_features):
        unique_labels = torch.unique(support_labels)
        # query to support
        class_sim_q2s = cos_sim(target_features, support_features_g)  # [35, 8, 5]
        class_dists_q2s = 1 - class_sim_q2s
        class_dists_q2s = [torch.sum(torch.sum(torch.index_select(class_dists_q2s, 2, extract_class_indices(support_labels, c)), dim=2), dim=1) for c in unique_labels]  # 每一帧的余弦相似度相加起来
        class_dists_q2s = torch.stack(class_dists_q2s).squeeze()
        if hasattr(self.args.MODEL, "USE_CONTRASTIVE") and self.args.MODEL.USE_CONTRASTIVE:
            class_dists_q2s = rearrange(class_dists_q2s * self.scale, 'c q -> q c')
        return class_dists_q2s


    def text_eh_temporal_transformer(self, context_support, support_features,target_features, support_labels):
        ######################################################
        token = context_support.mean(0) # (1, 1, 1024)
        support_token = self.token_tr(token, support_features)   # (25, 1, 1024)
        target_token = self.token_tr(token, target_features)   # (20, 1, 1024)
        ######################################################
        target_bs = target_features.shape[0]
        support_bs = support_features.shape[0]

        #一致性真实target token, 只有训练的时候用,元测试阶段其实可以没有这段代码，放在这里为了debug的时候，可以输出真实对比
        #沿着frame方向扩真，然后叠加到帧特征上
        s_number = support_features.shape[1]
        target_context_support_repeat = self.target_context_support.repeat(1, s_number, 1)
        target_features_t = target_features + 0.1 * target_context_support_repeat
        target_features_contra = torch.cat([self.target_context_support, target_features_t], dim=1)
        target_features_contra = self.context2(target_features_contra, target_features_contra, target_features_contra)

        #一致性的fake support token
        #support_token = token.expand(support_bs, -1, -1)
        #沿着frame方向扩真，然后叠加到帧特征上
        #s_number = support_features.shape[1]
        #support_token_repeat = support_token.repeat(1, s_number, 1)
        #support_features_t = support_features + 0.1 * support_token_repeat
        #support_features_contra = torch.cat([support_token, support_features_t], dim=1)
        #support_features_contra = self.context2(support_features_contra, support_features_contra, support_features_contra)
        support_features_contra = 0

        #对比比较的fake target
        #target_token= token.expand(target_bs, -1, -1)
        #沿着frame方向扩真，然后叠加到帧特征上
        s_number = support_features.shape[1]
        target_token_repeat = target_token.repeat(1, s_number, 1)
        target_features = target_features + 0.1 * target_token_repeat
        target_features = torch.cat([target_token, target_features], dim=1)
        target_features = self.context2(target_features, target_features, target_features)

        context_support = self.mid_layer(context_support)
        '''这里的MERGE_BEFORE参数是说，是先按类别聚合之后，再进行原型调制增强，还是先对样本进行调整增强之后再进行聚合 '''
        if hasattr(self.args.MODEL, "MERGE_BEFORE") and self.args.MODEL.MERGE_BEFORE:
            unique_labels = torch.unique(support_labels)
            support_features = [
                torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for
                c in unique_labels]
            support_features = torch.stack(support_features)
            context_support = [
                torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for
                c in unique_labels]
            context_support = torch.stack(context_support)
        #沿着frame方向扩真，然后叠加到帧特征上 增加context叠加

        #对比比较的real support
        s_number = support_features.shape[1]
        context_support_repeat = context_support.repeat(1, s_number, 1)
        support_features = support_features + 0.1 * context_support_repeat
        #support实际第一位用的是实际文本
        support_features = torch.cat([context_support,support_features], dim=1)  # 对应于R^txC 并上R^C--> R^(t+1)xC, 我们将文本特征沿着时间维度堆叠到相应的视频 Prototype modulation
        support_features = self.context2(support_features, support_features, support_features) # 对Prototype modulation之后的support进行temporal transformer操作

        if hasattr(self.args.MODEL, "MERGE_BEFORE") and self.args.MODEL.MERGE_BEFORE:
            pass
        else:
            unique_labels = torch.unique(support_labels)
            support_features_pro = [
                torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                for c in unique_labels]
            support_features_pro = torch.stack(support_features_pro)
        return support_features, target_features, support_features_pro, support_features_contra, target_features_contra, target_token, support_token

    def video2imagetext_adapter_mean(self, support_features, target_features):
        if self.training:
            text_features = self.text_features_train
        else:
            text_features = self.text_features_test
        # 是否使用分类标签 也就是文本分支
        if hasattr(self.args.MODEL, "USE_CLASSIFICATION") and self.args.MODEL.USE_CLASSIFICATION:
            feature_classification_in = torch.cat([support_features, target_features],  dim=0)  # 第一维度是support query一共的样本数量
            feature_classification = self.classification_layer(feature_classification_in).mean(1)  # 在第二位 帧的维度进行平均  维度为 [样本数量，1024] 公式2中的GAP操作
            class_text_logits = cos_sim(feature_classification, text_features) * self.scale  # 10 64， 10是10个视频样本，64是64个标签，对应于公式2
        else:
            class_text_logits = None
        return class_text_logits

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        gpus_use_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if self.args.DEVICE.NUM_GPUS > 1:
            self.backbone.cuda()
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(gpus_use_number)])


class token_trans(nn.Module):
    def __init__(self,):
        super().__init__()
        #self.trans = Transformer_v1(dim=1024, heads=8, dim_head_k=256, dropout_atte=0.2)
        self.mlp = FeedForward(1024, 2048, dropout=0.05)
    
    def forward(self, t, qu):
        # (1, 1, 1024) (20, 8, 1024)
        t = t.expand(qu.size(0), -1, -1)    # (20, 1, 1024)
        #x = self.trans(t, qu, qu)
        #x = self.mlp(x)
        x = self.mlp(t*qu.mean(dim=[1,2], keepdim=True))
        return x

def g_dis_c(support_features_g, support_labels, target_features_g):
        support_features_g = support_features_g.squeeze(1)
        target_features_g = target_features_g.squeeze(1)
        unique_labels = torch.unique(support_labels)
        # query to support
        class_sim_q2s = cos_sim(target_features_g, support_features_g)  # [20, 25]
        class_dists_q2s = 1 - class_sim_q2s
        class_dists_q2s = [torch.mean(torch.index_select(class_dists_q2s, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]  # 每一帧的余弦相似度相加起来
        class_dists_q2s = torch.stack(class_dists_q2s)
        class_dists_q2s = rearrange(class_dists_q2s, 'c q -> q c')
        return class_dists_q2s
