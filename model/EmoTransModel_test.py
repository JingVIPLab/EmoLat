from model.vgg import vgg
from model.decoder import decoder
from model.function import normal,normal_style
from model.function import calc_mean_std
import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
from model.ViT_helper import DropPath, to_2tuple, trunc_normal_
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from transformers import BertTokenizer, BertModel
from template import imagenet_templates
from torchvision import transforms
import numpy as np

class MLP(nn.Module):
    def __init__(self, num_fc_layers=1, in_dim=1536, h_dim=1024, o_dim=512, device='cpu', dtype=torch.float32):
        """
        初始化MLP类
        :param num_fc_layers: 全连接层的数量
        :param need_ReLU: 是否需要ReLU激活函数
        :param need_LN: 是否需要LayerNorm层
        :param need_Dropout: 是否需要Dropout层
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()  # 用于存储所有的层
        self.device = device 
        self.dtype = dtype

        input_dim = in_dim  # 假设输入特征维度为128
        hidden_dim = h_dim  # 每层隐藏单元的数量

        for i in range(num_fc_layers):
            # 添加全连接层
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=0.5))
            # 下一层的输入维度为当前隐藏单元的数量
            input_dim = hidden_dim

        # 输出层
        self.layers.to(self.device, self.dtype)
        self.output_layer = nn.Linear(hidden_dim, o_dim).to(self.device, self.dtype)  # 假设输出是标量
        self.to(self.device, self.dtype)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return: 模型输出
        """
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class Transform(nn.Module):
    def __init__(self, in_planes,device,dtype=torch.float32):
        super(Transform, self).__init__()
        self.device = device
        self.dtype = dtype
        self.sanet4_1 = SANet(in_planes=in_planes).to(self.device).to(self.dtype)
        # self.sanet5_1 = SANet(in_planes=in_planes)
        # self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1)).to(self.device).to(self.dtype)
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3)).to(self.device).to(self.dtype)

    def forward(self, content4_1, style4_1):
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1)))

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        return x

        

class EmoTransModel_test(nn.Module):
    def __init__(
                self, 
                encoder, # load
                decoder, # load
                PatchEmbed, # load
                transformer, # load
                embedding_group, # load
                text_encoder_clip, # load
                emo_conv, # load
                Mapper, # load
                device, 
                dtype=torch.float32):
        super().__init__()

        self.dtype = dtype
        self.device = device
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.enc_5 = nn.Sequential(*enc_layers[31:44])

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.transformer = transformer
        hidden_dim = transformer.d_model       
        self.decoder = decoder
        self.embedding = PatchEmbed
        self.embedding_group = embedding_group
        self.text_encoder_clip = text_encoder_clip
        self.emo_conv = emo_conv
        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        self.Mapper = Mapper.to(self.device).to(self.dtype)
        for param in self.Mapper.parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    
    def standardize_embeddings(self, emotion_key):
        # 获取当前情感类别的嵌入权重
        embedding_weights = self.embedding_group[emotion_key].weight.detach().cpu().numpy()

        # 计算嵌入的均值和标准差
        mean_value = embedding_weights.mean(axis=1)[0]  # 计算均值（按列）
        std_value = np.var(embedding_weights, axis=0)    # 计算标准差（按列）

        # 对嵌入进行标准化 (减去均值然后除以标准差)
        standardized_embedding_weights = (embedding_weights - mean_value) / (std_value + 1e-10)
        
        return standardized_embedding_weights

    def sample_from_standardized_embeddings(self, emotion_key):
        # 标准化嵌入权重
        standardized_weights = self.standardize_embeddings(emotion_key)
        
        # 将标准化后的权重转换为torch张量
        standardized_weights_tensor = torch.tensor(standardized_weights, dtype=self.dtype)
        
        # 随机选择一个嵌入的索引
        random_index = torch.randint(0, standardized_weights_tensor.size(0), (1,))
        
        # 根据选择的索引返回对应的嵌入
        sampled_embedding = standardized_weights_tensor[random_index]  # 扩展维度使其匹配输入的形状
        
        return sampled_embedding

    def selent_from_standardized_embeddings_index(self, emotion_key, index):
        # 标准化嵌入权重
        standardized_weights = self.standardize_embeddings(emotion_key)
        
        # 将标准化后的权重转换为torch张量
        standardized_weights_tensor = torch.tensor(standardized_weights, dtype=self.dtype)
        
        # 随机选择一个嵌入的索引
        # random_index = torch.randint(0, standardized_weights_tensor.size(0), (1,))
        
        # 根据选择的索引返回对应的嵌入
        index = torch.tensor([index]).to(self.device)
        sampled_embedding = standardized_weights_tensor[index]  # 扩展维度使其匹配输入的形状
        
        return sampled_embedding


    def forward(self, samples_c, text_label, emb_index=-1):

        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
        with torch.no_grad():
            text_label_clip = text_label
            text_tokens = clip.tokenize(text_label_clip).to(self.device)
            text_feats_clip = self.text_encoder_clip.encode_text(text_tokens).to(self.device)
    
        text_feats = text_feats_clip
        item = text_label.lower()
        if emb_index == -1:
            emb_vec = self.sample_from_standardized_embeddings(item).to(self.device).to(self.dtype)
        else:
            # import pdb;pdb.set_trace()
            emb_vec = self.selent_from_standardized_embeddings_index(item, emb_index).to(self.device).to(self.dtype)
        concat_feats = torch.cat([text_feats, emb_vec], dim=1).to(self.device)
        concat_feats = self.Mapper(concat_feats)
        concat_feats = concat_feats.unsqueeze(-1).unsqueeze(-1)
        concat_feats = concat_feats.repeat(1, 1, 16, 16)
        text_feats = self.emo_conv(concat_feats) # b,512,32,32

        content = self.embedding(samples_c.tensors) # pathembed

        pos_s = None
        pos_c = None
        mask = None
        fusion = self.transformer(text_feats, mask, content, pos_c, pos_s) 
        fusion_img = self.decoder(fusion)

        return fusion_img
