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

        

class EmoTransModel(nn.Module):
    def __init__(
                self, 
                encoder,
                decoder, 
                PatchEmbed, 
                transformer, 
                embedding_group, 
                emb_means, 
                emb_std,
                text_encoder_clip, 
                graph_encoder, 
                clip_emo_classifier, 
                discriminator,
                sentiment_vector,
                emo_conv,
                Mapper,
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

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model       
        self.decoder = decoder
        self.embedding = PatchEmbed
        self.embedding_group = embedding_group
        self.means = emb_means
        self.std = emb_std

        self.text_encoder_clip = text_encoder_clip
        self.graph_encoder = graph_encoder

        self.clip_emo_classifier = clip_emo_classifier
        self.discriminator = discriminator

        self.emo_conv = emo_conv

        conv1, conv2, conv3, conv4, conv5 = sentiment_vector
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            for param in getattr(self,name).parameters():
                param.requires_grad = False
        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)

        self.distance_l2 = torch.nn.PairwiseDistance(p=2).to(device)
        self.adversarial_loss = torch.nn.MSELoss()

        self.Mapper = Mapper.to(self.device).to(self.dtype)
        for param in self.Mapper.parameters():
                param.requires_grad = False

        # self.discriminator = Discriminator(device=self.device)

        self.cropper = transforms.Compose([
            transforms.RandomCrop(64)
        ])
        self.augment = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
            transforms.Resize(224)
        ])

        self.emotion_dict = {
            "amusement":torch.tensor([1.,0,0,0,0,0,0,0]).to(self.device).to(self.dtype),
            "anger":torch.tensor([0,1.,0,0,0,0,0,0]).to(self.device).to(self.dtype),
            "awe":torch.tensor([0,0,1.,0,0,0,0,0]).to(self.device).to(self.dtype),
            "contentment":torch.tensor([0,0,0,1.,0,0,0,0]).to(self.device).to(self.dtype),
            "disgust":torch.tensor([0,0,0,0,1.,0,0,0]).to(self.device).to(self.dtype),
            "excitement":torch.tensor([0,0,0,0,0,1.,0,0]).to(self.device).to(self.dtype),
            "fear":torch.tensor([0,0,0,0,0,0,1.,0]).to(self.device).to(self.dtype),
            "sadness":torch.tensor([0,0,0,0,0,0,0,1.]).to(self.device).to(self.dtype)
            }

        self.celoss = nn.CrossEntropyLoss().to(self.device).to(self.dtype)

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def TV_loss(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

    def cal_SV(self,content_feats):
        gram_pre_1 = self.conv1(content_feats[0]).flatten(2)
        gram_pre_2 = self.conv2(content_feats[1]).flatten(2)
        gram_pre_3 = self.conv3(content_feats[2]).flatten(2)
        gram_pre_4 = self.conv4(content_feats[3]).flatten(2)
        gram_pre_5 = self.conv5(content_feats[4]).flatten(2)
        gram_pre_1_T = gram_pre_1.transpose(1, 2)
        gram_pre_2_T = gram_pre_2.transpose(1, 2)
        gram_pre_3_T = gram_pre_3.transpose(1, 2)
        gram_pre_4_T = gram_pre_4.transpose(1, 2)
        gram_pre_5_T = gram_pre_5.transpose(1, 2)
        gram_1 = gram_pre_1.bmm(gram_pre_1_T).reshape(-1,256)
        gram_2 = gram_pre_2.bmm(gram_pre_2_T).reshape(-1,256)
        gram_3 = gram_pre_3.bmm(gram_pre_3_T).reshape(-1,256)
        gram_4 = gram_pre_4.bmm(gram_pre_4_T).reshape(-1,256)
        gram_5 = gram_pre_5.bmm(gram_pre_5_T).reshape(-1,256)
        gram_content_1 = torch.nn.functional.normalize(gram_1, p=2, dim=1)
        gram_content_2 = torch.nn.functional.normalize(gram_2, p=2, dim=1)
        gram_content_3 = torch.nn.functional.normalize(gram_3, p=2, dim=1)
        gram_content_4 = torch.nn.functional.normalize(gram_4, p=2, dim=1)
        gram_content_5 = torch.nn.functional.normalize(gram_5, p=2, dim=1)
        gram_content = torch.cat([gram_content_1,gram_content_2,gram_content_3,gram_content_4,gram_content_5],dim=0)
        return gram_content

    def cal_homo_loss(self, content_feats, style_feats):
        content_SV = self.cal_SV(content_feats)
        style_SV = self.cal_SV(style_feats)
        gram_loss = self.distance_l2(content_SV,style_SV)
        return gram_loss.sum()
    
    def Triplet_loss(self, anchor, pos, neg, rel, margin1 = 0.2, margin2 = 0.1):
        dis_ap = self.distance_l2(anchor,pos)
        dis_ar = 0.5 * self.distance_l2(anchor,rel)
        dis_an = 0.2 * self.distance_l2(anchor,neg)
        loss_1 = dis_ap - dis_ar + margin1
        loss_2 = dis_ar - dis_an + margin2
        zeros = torch.zeros(loss_1.shape).to(self.device)
        loss_1 = torch.maximum(loss_1,zeros)
        loss_2 = torch.maximum(loss_2,zeros)
        loss_1 = loss_1.sum()
        loss_2 = loss_2.sum()
        return loss_1 + loss_2
    
    def all_text_label_to_template(self):
        text_template = "a photo seems like "
        new_text_label_batch = []
        for item in ['amusement','anger','awe','contentment','disgust','excitement','fear','sadness']:
            temp = text_template
            item = "".join([text_template, item])
            new_text_label_batch.append(item)
        return new_text_label_batch
    
    def text_label_to_template(self, text_label_batch):
        text_template = "a photo seems like "
        new_text_label_batch = []
        for item in text_label_batch:
            temp = text_template
            item = "".join([text_template, item[0]])
            new_text_label_batch.append(item)
        return new_text_label_batch
    
    def text_label_to_index(self, text_label_batch):
        # import pdb;pdb.set_trace()
        emotions = [['amusement'],['anger'],['awe'],['contentment'],['disgust'],['excitement'], ['fear'],['sadness']]
        indices = [emotions.index(label) for label in text_label_batch]
        return indices


    # amu awe anger contentment disgust fear excite sadness
    def clip_loss(self, gen_img, text_label_batch):
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            image_features = self.clip_emo_classifier.encode_image(gen_img.to(self.device))
        new_text_label_batch = self.text_label_to_template(text_label_batch)
        all_text_label_batch = self.all_text_label_to_template()
        with torch.no_grad():
            text_tokens = clip.tokenize(all_text_label_batch).to(self.device)
            text_features = self.clip_emo_classifier.encode_text(text_tokens)
        # Normalize image and text features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Calculate similarity scores
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        indices = self.text_label_to_index(text_label_batch)
        label_tensor = []
        for text in text_label_batch:
            label_tensor.append(self.emotion_dict[text[0]])
        # import pdb;pdb.set_trace()
        label_tensor = torch.stack(label_tensor)
        label_loss = self.celoss(similarity, label_tensor)
        return label_loss


        # values = []
        # clip_loss_value = 0
        # for i, item in enumerate(similarity):
        #     clip_loss_value += item[indices[i]]
        # return clip_loss_value

    
    def graph_loss(self, gen_img, graph_data):
        # 训练好的鉴别器 判断生图和图结构 的 相似性

        b,_,_,_ = gen_img.shape        
        Tensor = torch.FloatTensor
        real_label = Tensor(b,1).fill_(0.9).to(self.device).to(self.dtype).detach()
        fake_label = Tensor(b,1).fill_(0.1).to(self.device).to(self.dtype).detach()
        with torch.no_grad():
            graph_feats = self.graph_encoder(graph_data).to(self.dtype)
            real_loss = self.adversarial_loss(self.discriminator(graph_feats), real_label).to(self.dtype)
            fake_loss = self.adversarial_loss(self.discriminator(gen_img), fake_label).to(self.dtype)
        loss_graph = (real_loss + fake_loss) / 2
        return loss_graph

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]
    
    def clip_normalize(self, image):
        image = F.interpolate(image,size=224,mode='bicubic')
        mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device).to(self.dtype)
        std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device).to(self.dtype)
        mean = mean.view(1,-1,1,1)
        std = std.view(1,-1,1,1)

        image = (image-mean)/std
        return image

    def get_image_prior_losses(self, inputs_jit):
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        
        return loss_var_l2

    def directional_loss(self, prompt, content_image, target, num_crops=4):
        source = "a photo "
        with torch.no_grad():
            template_text = self.compose_text_with_templates(prompt, imagenet_templates)
            tokens = clip.tokenize(template_text).to(self.device)
            text_features = self.clip_model.encode_text(tokens).detach()
            text_features = text_features.mean(axis=0, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            template_source = self.compose_text_with_templates(source, imagenet_templates)
            tokens_source = clip.tokenize(template_source).to(self.device)
            text_source = self.clip_model.encode_text(tokens_source).detach()
            text_source = text_source.mean(axis=0, keepdim=True)
            text_source /= text_source.norm(dim=-1, keepdim=True)
            source_features = self.clip_model.encode_image(self.clip_normalize(content_image))
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

        loss_patch=0
        img_proc =[]
        for n in range(num_crops):
            target_crop = self.cropper(target)
            target_crop = self.augment(target_crop)
            img_proc.append(target_crop)

        img_proc = torch.cat(img_proc, dim=0)
        img_aug = img_proc

        image_features = self.clip_model.encode_image(self.clip_normalize(img_aug))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
        
        img_direction = (image_features-source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
        
        text_direction = (text_features-text_source).repeat(image_features.size(0),1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
        loss_temp[loss_temp < 0.7] = 0
        loss_patch += loss_temp.mean()
        
        glob_features = self.clip_model.encode_image(self.clip_normalize(target))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
        
        glob_direction = (glob_features - source_features)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
        
        loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()
        
        lambda_tv = 0.5
        lambda_patch = 0.5
        reg_tv = lambda_tv * self.get_image_prior_losses(target)

        return reg_tv + loss_glob + loss_patch * lambda_patch
    



    def cal_loss(self, samples_c, samples_s, text_label, graph_data, fusion_img_feats,fusion_img, content_feats, style_feats, text_feats,style_embedding):
        s_mean = (0.52, 0.465, 0.40)
        # s_mean = (0.485,0.456,0.406)
        s_std = (0.22, 0.21,0.19)
        # s_std = (0.229,0.224,0.225)
        mask = None
        pos_c = None
        pos_s = None
        loss_c = self.calc_content_loss(normal(fusion_img_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(fusion_img_feats[-2]), normal(content_feats[-2]))
        loss_s = self.calc_style_loss(fusion_img_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(fusion_img_feats[i], style_feats[i])
        Iss = self.decoder(self.transformer(text_feats, mask , style_embedding, pos_s, pos_s))     
 
        loss_lambda1 = self.calc_content_loss(Iss, samples_s.tensors)
        Iss_feats=self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Iss_feats[i], style_feats[i])
        loss_tv = self.TV_loss(fusion_img)
        std_s = torch.tensor(s_std).view(1, -1, 1, 1).to(self.device)
        mean_s = torch.tensor(s_mean).view(1, -1, 1, 1).to(self.device)

        Ics_denorm = fusion_img * (std_s) + (mean_s)
        style_denorm = samples_s.tensors # * std_s + mean_s
        Ics_feats_denorm = self.encode_with_intermediate(Ics_denorm)
        style_feats_denorm = self.encode_with_intermediate(style_denorm)
        loss_homo = self.cal_homo_loss(Ics_feats_denorm,style_feats_denorm)
        # Ics_denorm_clip = Ics_denorm
        Ics_denorm_clip = F.interpolate(Ics_denorm, size=(224, 224), mode='bilinear', align_corners=False)
        # loss_clip = self.clip_loss(Ics_denorm_clip, text_label)
        # loss_graph = self.graph_loss(Ics_feats_denorm[3], graph_data)
        loss_graph = 0
        loss_directional = 0
        for index, prompt in enumerate(text_label):
            loss_directional += self.directional_loss(prompt, samples_c.tensors[index].unsqueeze(dim=0), Ics_denorm[index].unsqueeze(dim=0))
    
        loss_c = loss_c
        loss_s = loss_s
        loss_lambda1 = loss_lambda1
        loss_lambda2 = loss_lambda2
        loss_tv = loss_tv
        loss_homo = 20 * (loss_homo)
        # loss_label = loss_label
        


        return Ics_denorm, loss_c, loss_s, loss_lambda1, loss_lambda2, loss_tv, loss_homo, _, loss_graph, loss_directional

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
    

    def forward(self, samples_c, samples_s, text_label, graph_data):

        text = [item[0] for item in text_label]
        # import pdb;pdb.set_trace()
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s) 
        with torch.no_grad():
            text_label_clip = [item[0] for item in text_label]
            text_tokens = clip.tokenize(text_label_clip).to(self.device)
            text_feats_clip = self.text_encoder_clip.encode_text(text_tokens).to(self.device)
        

        latent_feats = []
        for index, item in enumerate(text):
            # text clip feats
            text_feats = text_feats_clip[index]
            text_feats = text_feats.unsqueeze(0)
            item = item.lower()

            # 情感空间 采样
            # import pdb;pdb.set_trace()
            emb_vec = self.sample_from_standardized_embeddings(item).to(self.device).to(self.dtype)
            # emb_vec = self.sample_from_embedding_with_stats(item, 10) # (n, 1024) [n,1024,1,1]
            concat_feats = torch.cat([text_feats, emb_vec], dim=1).to(self.device)
            concat_feats = self.Mapper(concat_feats)
            concat_feats = concat_feats.unsqueeze(-1).unsqueeze(-1)
            concat_feats = concat_feats.repeat(1, 1, 16, 16)
            latent_feats.append(concat_feats)
        latent_feats = torch.cat(latent_feats,dim=0)
        text_feats = latent_feats
    
        content_feats = self.encode_with_intermediate(samples_c.tensors)  # extract content relu1_1-4_1
        style_feats = self.encode_with_intermediate(samples_s.tensors)  # extract style relu1_1-4_1 # 做损失用的
        text_feats = self.emo_conv(text_feats) # b,512,32,32
        
        # import pdb;pdb.set_trace()
        content = self.embedding(samples_c.tensors) # pathembed
        style_embedding = self.embedding(samples_s.tensors)
        
        pos_s = None
        pos_c = None
        mask = None
        fusion = self.transformer(text_feats, mask, content, pos_c, pos_s) 
        fusion_img = self.decoder(fusion)
        fusion_img_feats = self.encode_with_intermediate(fusion_img)

        loss_afi = self.cal_loss(samples_s, 
                      samples_c, 
                      text_label,
                      graph_data,
                      fusion_img_feats,
                      fusion_img, 
                      content_feats, 
                      style_feats, 
                      text_feats,
                      style_embedding
                      )

        Ics_denorm, loss_c, loss_s, loss_lambda1, loss_lambda2, loss_tv, loss_homo, _, loss_graph, loss_directional = loss_afi
        
        # loss_clip = self.clip_loss(fusion_img, text_label)
        # return Ics_denorm, loss_c, loss_s, loss_lambda1, loss_lambda2, loss_tv, loss_homo, loss_clip, loss_graph, loss_directional
        GD = (style_feats, fusion_img_feats, text_feats_clip)
        return Ics_denorm, loss_c, loss_s, loss_lambda1, loss_lambda2, loss_tv, loss_homo, _, 0, loss_directional, GD