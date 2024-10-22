import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .myRes import extract_class_indices, cosine_dist


class CNN(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

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
            # from .myRes import resnet50_2
            # resnet = resnet50_2(weights=models.ResNet50_Weights.DEFAULT)
        self.ddd = True

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        #self.resnet1 = nn.Sequential(*list(resnet.children())[:-4])
        #self.resnet2 = nn.Sequential(*list(resnet.children())[-4:-1])
        #self.man = GroupGLKA(3)
        from .myRes import mo_3
        self.mo = mo_3()
        self.avg=nn.AdaptiveAvgPool2d(1)
    def forward(self, context_images, context_labels, target_images):

        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''
        
        # o1_s = self.man(context_images)
        # o1_q = self.man(target_images)
        
        context_features = self.resnet(context_images) # 200 x 2048
        target_features = self.resnet(target_images) # 160 x 2048
        mo_logits = self.mo(target_features, context_features, context_labels)
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = self.avg(context_features) # 200 x 2048
        target_features = self.avg(target_features)
        context_features = context_features.reshape(-1, self.cfg.DATA.SEQ_LEN, self.cfg.trans_linear_in_dim) # 25 x 8 x 2048
        target_features = target_features.reshape(-1, self.cfg.DATA.SEQ_LEN, self.cfg.trans_linear_in_dim) # 20 x 8 x 2048

        unique_labels = torch.unique(context_labels)
        if self.ddd:
            for c in unique_labels:
                print(extract_class_indices(context_labels, c))
            self.ddd=False
        su = [context_features[extract_class_indices(context_labels, c)] for c in unique_labels] # 5 5x8x2048
        su = torch.stack(su, dim=0)
        su = su.mean(1).mean(1) # 5x2048
        qu = target_features.mean(1) # 20x2048
            
        dist = cosine_dist(qu,su) 
        #print(dist.shape)
        probability = torch.nn.functional.softmax(dist, dim=-1)
        
        return_dict = {'logits': probability.unsqueeze(0), 
                       'mo_logits': mo_logits,
                       }

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

######################################################################



