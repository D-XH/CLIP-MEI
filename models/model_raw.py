import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

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
            from myRes import resnet50_2
            # resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            resnet = resnet50_2(weights=models.ResNet50_Weights.DEFAULT)
        self.ddd = True

        last_layer_idx = -1
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        #self.resnet1 = nn.Sequential(*list(resnet.children())[:-4])
        #self.resnet2 = nn.Sequential(*list(resnet.children())[-4:-1])
        #self.man = GroupGLKA(3)
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

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
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
        
        return_dict = {'logits': probability.unsqueeze(0)}

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

def cosine_dist(x,y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    cosine_sim_list = []
    for i in range(m):
        y_tmp = y[i].unsqueeze(0)
        x_tmp = x
        #print(x_tmp.size(),y_tmp.size())
        cosine_sim = nn.functional.cosine_similarity(x_tmp,y_tmp)
        cosine_sim_list.append(cosine_sim)
    return torch.stack(cosine_sim_list).transpose(0,1)

############### big kernel multi scale conv
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class GroupGLKA(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2*n_feats
        
        self.n_feats= n_feats
        self.i_feats = i_feats
        
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        #Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 7, 1, 7//2, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 9, stride=1, padding=(9//2)*4, groups=n_feats//3, dilation=4),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 7, stride=1, padding=(7//2)*3, groups=n_feats//3, dilation=3),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 5, stride=1, padding=(5//2)*2, groups=n_feats//3, dilation=2),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        
        self.X3 = nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3)
        self.X5 = nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3)
        self.X7 = nn.Conv2d(n_feats//3, n_feats//3, 7, 1, 7//2, groups= n_feats//3)
        
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        
    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()
        
        x = self.norm(x)
        
        x = self.proj_first(x)
        
        a, x = torch.chunk(x, 2, dim=1) 
        
        a_1, a_2, a_3= torch.chunk(a, 3, dim=1)
        
        a = torch.cat([self.LKA3(a_1)*self.X3(a_1), self.LKA5(a_2)*self.X5(a_2), self.LKA7(a_3)*self.X7(a_3)], dim=1)
        
        x = self.proj_last(x*a)*self.scale + shortcut
        
        return x  