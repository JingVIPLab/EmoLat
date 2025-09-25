import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch_geometric.nn import GCNConv
from collections import OrderedDict
# from emo_token_gen_model.vgg import vgg as vgg
from tqdm import tqdm
import math


MSE_Loss = nn.MSELoss()

clip_version = "ViT-B/32"

# Clip Image Encoder
class ClipImageEncoder(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(ClipImageEncoder, self).__init__()
        self.device = device
        self.dtype = dtype
        self.clip_model, _ = clip.load(clip_version, self.device)
        # 冻结 CLIP 模型的所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
    def scale_feat(self, x):
        # 计算每个特征的最小值和最大值
        x_min = x.min(dim=1, keepdim=True)[0]  # 计算每个样本的最小值
        x_max = x.max(dim=1, keepdim=True)[0]  # 计算每个样本的最大值

        # 将特征归一化到 [0, 1]
        x_normalized = (x - x_min) / (x_max - x_min + 1e-8)  # 避免除以零

        # 将特征缩放到 [-1, 1]
        x_scaled = x_normalized * 2 - 1
        return x_scaled

    def forward(self, images):
        #  import pdb;pdb.set_trace()
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images.to(self.device))
        image_features.to(self.device)# [B,512]
        # norm to [-1, 1]
        # image_features = self.scale_feat(image_features)

        return image_features.to(self.dtype)

# Clip Text Encoder
class ClipTextEncoder(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(ClipTextEncoder, self).__init__()
        self.device = device
        self.dtype = dtype
        self.clip_model, _ = clip.load(clip_version, self.device)
        # 冻结 CLIP 模型的所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
    def forward(self, text):
        # 这里 text 是一个包含多个文本字符串的列表
        with torch.no_grad():
            # 将文本转换为 tokens
            text_tokens = clip.tokenize(text).to(self.device)
            # 使用 CLIP 模型编码文本
            text_features = self.clip_model.encode_text(text_tokens)
        
        # 返回编码后的文本特征，大小为 [B, 512]
        return text_features.to(self.dtype)

def load_model(model_struct, model_name, device):
    print("loading vgg")
    # 假设 args.emo_encoder 是模型权重文件的路径
    weights = torch.load(model_name,map_location=device)

    # 获取权重的总数
    total_weights = sum([param.numel() for param in weights.values()])

    # 初始化已加载的权重计数
    loaded_weights = 0

    # 遍历权重字典并逐个加载
    name = model_name.split('/')[-1]
    for key, value in tqdm(weights.items(), total=len(weights), desc=f'Loading {name} weights'):
        if key in model_struct.state_dict():
            model_struct.state_dict()[key].copy_(value)
            loaded_weights += value.numel()

    model_struct.to(device)
    print("load vgg successful")
    return model_struct

class ImageEncoder(nn.Module):
    def __init__(self, vgg, device, dtype=torch.float32):
        super(ImageEncoder, self).__init__()
        vgg = nn.Sequential(*list(vgg.children())[:44])
        vgg_path = "/home/zjx/lab/my3_lab/v1/pretrained_model/vgg_normalised.pth"
        vgg = load_model(vgg, vgg_path, device)
        encoder = vgg
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def forward(self, imgs):
        imgs = self.encode_with_intermediate(imgs)  # 计算style的特征
        return imgs[4] # [B,512,32,32]
        
# Emotion Forest Encoder (Graph Convolution Network)
class GraphEncoder(nn.Module):
    def __init__(self, do, da, dr, ds, dg, device, clip_model_name=clip_version, dtype=torch.float32):
        super(GraphEncoder, self).__init__()
        
        # set some args
        self.dg = dg

        # set device
        self.device = device
        self.dtype = dtype

        # Load CLIP model for text feature extraction
        self.clip_model, _ = clip.load(clip_model_name, device=self.device)
        # 冻结 CLIP 模型的所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # MLP layers to map object and adjective nodes to the same dimension
        self.Fo_dg = nn.Sequential(
            nn.Linear(do, dg),
            # nn.ReLU(),
            # nn.Linear(dg, dg)
        ).to(self.device)
        self.Fa_dg = nn.Sequential(
            nn.Linear(da, dg),
            # nn.ReLU(),
            # nn.Linear(dg, dg)
        ).to(self.device)
        self.Fs_dg = nn.Sequential(
            nn.Linear(ds, dg),
            # nn.ReLU(),
            # nn.Linear(dg, dg)
        ).to(self.device)
        

        # QKV layers for attention mechanism
        self.query_layer = nn.Linear(2 * dg, dg).to(dtype=self.dtype).to(self.device)
        self.key_layer = nn.Linear(2 * dg, dg).to(dtype=self.dtype).to(self.device)
        self.value_layer = nn.Linear(2 * dg, dg).to(dtype=self.dtype).to(self.device)
        

        # Initialize GCN layers for object and relationship embeddings
        self.gconv_all = GCNConv(dr, 16 * 16).to(self.device)

        # emo and graph 
        self.Linear_emo_graph = nn.Sequential(
            # nn.Linear(2 * dg, 2 * dg),
            nn.Linear(2 * dg, dg),
            nn.BatchNorm1d(dg),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dg, dg)
        ).to(self.device)

        self.conv_emb = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=32,out_channels=512).to(self.device)

        # self.share_linear_block = share_linear_block
        for param in self.Fo_dg.parameters():
            param.requires_grad = True
        for param in self.Fa_dg.parameters():
            param.requires_grad = True
        for param in self.Fs_dg.parameters():
            param.requires_grad = True
        for param in self.query_layer.parameters():
            param.requires_grad = True
        for param in self.key_layer.parameters():
            param.requires_grad = True
        for param in self.value_layer.parameters():
            param.requires_grad = True
        for param in self.gconv_all.parameters():
            param.requires_grad = True
        for param in self.Linear_emo_graph.parameters():
            param.requires_grad = True

    def create_graph_all(self, object_words_batch, adjective_words_batch, global_adj_word_batch):
        batch_size = len(object_words_batch)
        edges_batch = []
        total_words_batch = []
        adjs_batch = []
        
        for b in range(batch_size):
            object_words = object_words_batch[b]
            adjective_words = adjective_words_batch[b]
            global_adj_word = global_adj_word_batch[b]

            set_adjs = set(adjective_words + global_adj_word)
            total_words = list(OrderedDict.fromkeys(object_words + adjective_words + global_adj_word))
            total_words_batch.append(total_words)
            adjs_batch.append(set_adjs)

            word_to_index = {word: idx for idx, word in enumerate(total_words)}

            edges = set()

            for obj_word, adj_word in zip(object_words, adjective_words):
                from_idx = word_to_index[obj_word]
                to_idx = word_to_index[adj_word]
                edges.add((from_idx, to_idx))

            global_adj = global_adj_word[0]
            global_adj_idx = word_to_index[global_adj]
            for obj_word in object_words[:len(adjective_words)]:
                from_idx = word_to_index[obj_word]
                to_idx = global_adj_idx
                edges.add((from_idx, to_idx))

            new_edges = []
            total_words_extended = total_words.copy()
            for from_idx, to_idx in edges:
                new_node_idx = len(total_words_extended)
                total_words_extended.append(f"{from_idx}_{to_idx}")
                new_edges.append((from_idx, new_node_idx))
                new_edges.append((to_idx, new_node_idx))
                new_edges.append((from_idx, to_idx))

            edges_from = [edge[0] for edge in new_edges]
            edges_to = [edge[1] for edge in new_edges]
            edge_index = torch.tensor([edges_from, edges_to], dtype=torch.long).to(self.device)
            edges_batch.append(edge_index)

        return total_words_batch, edges_batch, adjs_batch

    def get_relation_feat(self, object_words_batch, adjective_words_batch, global_adj_word_batch, sentiment_word_batch):
        # import pdb;pdb.set_trace()
        batch_size = len(object_words_batch)
        G_attention_rel_batch = []
        # clip_sentiment_features_batch = []
        b = 0

        # for b in range(batch_size):
            # import pdb;pdb.set_trace()
        object_words = object_words_batch[b]
        adjective_words = adjective_words_batch[b]
        global_adj_word = global_adj_word_batch[b]
        sentiment_word = sentiment_word_batch[b]

        text_obj_tokens = clip.tokenize(object_words).to(self.device)
        text_adj_tokens = clip.tokenize(adjective_words).to(self.device)
        text_g_adj_tokens = clip.tokenize(global_adj_word).to(self.device)
        text_sentiment_tokens = clip.tokenize(sentiment_word).to(self.device)

        with torch.no_grad():
            clip_obj_features = self.clip_model.encode_text(text_obj_tokens).to(self.dtype).to(self.device)
            clip_adj_features = self.clip_model.encode_text(text_adj_tokens).to(self.dtype).to(self.device)
            clip_g_adj_features = self.clip_model.encode_text(text_g_adj_tokens).to(self.dtype).to(self.device)
            clip_sentiment_features = self.clip_model.encode_text(text_sentiment_tokens).to(self.dtype).to(self.device)
            # clip_sentiment_features_batch.append(clip_sentiment_features)

        G_obj_dg = self.Fo_dg(clip_obj_features)
        G_adj_dg = self.Fa_dg(clip_adj_features)
        G_g_adj_dg = self.Fa_dg(clip_g_adj_features)
        G_sentiment_dg = self.Fs_dg(clip_sentiment_features)


        # import pdb;pdb.set_trace()
        G_rel_feat = torch.cat([G_obj_dg, G_adj_dg], dim=1)
        G_g_adj_repeat_dg = G_g_adj_dg.repeat(len(object_words), 1)
        G_rel_feat_1 = torch.cat([G_obj_dg, G_g_adj_repeat_dg], dim=1)
        G_rel_feat = torch.cat([G_rel_feat, G_rel_feat_1], dim=0)

        G_sentiment_cat_dg = torch.cat([G_sentiment_dg, G_sentiment_dg], dim=1)
        G_sentiment_repeat_dg = G_sentiment_cat_dg.repeat(2 * len(object_words), 1)

        query = self.query_layer(G_rel_feat)
        key = self.key_layer(G_sentiment_repeat_dg)
        value = self.value_layer(G_sentiment_repeat_dg)

        attention_scores = torch.matmul(query, key.T) / (self.dg ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        G_attention_rel = torch.matmul(attention_weights, value)

        G_attention_rel_batch.append(G_attention_rel)

        
        return torch.stack(G_attention_rel_batch), clip_sentiment_features
    
    def pad_to_14_rows(self, GCN_feat):
        num, out_dim = GCN_feat.shape
        if num < 32:
            # 计算需要补充的行数
            padding_rows = 32 - num
            # 获取最后一行并复制
            last_row = GCN_feat[-1].unsqueeze(0)  # [1, out_dim]
            padding = last_row.repeat(padding_rows, 1)  # [padding_rows, out_dim]
            # 拼接原始张量和补充的行
            GCN_feat = torch.cat([GCN_feat, padding], dim=0)  # [28, out_dim]
        elif num >= 16:
            GCN_feat = GCN_feat[:32]
        return GCN_feat

    def forward(self, data):
        object_words_batch, adjective_words_batch, sentiment_word_batch, global_adj_word_batch = data
        total_words_all, edge_index_all, adjs_all = self.create_graph_all(object_words_batch, adjective_words_batch, global_adj_word_batch)

        batch_size = len(object_words_batch)
        GCN_feat_batch = []

        clip_sentiment_features_batch = []

        for b in range(batch_size):
            total_words = total_words_all[b]
            edge_index = edge_index_all[b]

            num_objects = len(object_words_batch[b])
            num_adjectives = len(adjs_all[b])
            words_obj_adj = total_words[:num_adjectives + num_objects]
            
            if len(words_obj_adj) > 77:
                words_obj_adj = words_obj_adj[:77]

            text_tokens = clip.tokenize(words_obj_adj).to(self.device)
            with torch.no_grad():
                clip_features = self.clip_model.encode_text(text_tokens).to(self.dtype).to(self.device)

            G_obj = clip_features[:num_objects]
            G_adj = clip_features[num_objects:num_objects + num_adjectives]

            G_obj_dg = self.Fo_dg(G_obj)
            G_adj_dg = self.Fa_dg(G_adj)

            G_attention_rel, clip_sentiment_features = self.get_relation_feat([object_words_batch[b]], [adjective_words_batch[b]], [global_adj_word_batch[b]], [sentiment_word_batch[b]])
            clip_sentiment_features_batch.append(clip_sentiment_features)
            # import pdb;pdb.set_trace()

            G_global = torch.cat([G_obj_dg, G_adj_dg, G_attention_rel.squeeze(0)], dim=0)

            # import pdb;pdb.set_trace()
            try:
                GCN_feat = self.gconv_all(G_global, edge_index) # n_node, out_dim
            except Exception as e:
                print(edge_index)
            GCN_feat_padded = self.pad_to_14_rows(GCN_feat)  
            # GCN_feat = self.gconv_all(GCN_feat, edge_index)

            # GCN_feat_pooled = GCN_feat.mean(dim=0)
            # import pdb;pdb.set_trace()
            GCN_feat_batch.append(GCN_feat_padded) # [b, n_node, out_dim]

        # import pdb;pdb.set_trace()
        GCN_feat_batch = torch.stack(GCN_feat_batch, dim=0) # [b, n_node, out_dim] [b,28,1024]
        # GCN_feat_batch_ori = GCN_feat_batch.clone()
        # clip_sentiment_features_batch = torch.cat(clip_sentiment_features_batch,dim=0)
        # GCN_feat_batch = torch.cat([GCN_feat_batch, clip_sentiment_features_batch],dim=1)
        # GCN_feat_batch = torch.add(GCN_feat_batch, clip_sentiment_features_batch)
        # GCN_feat_batch = self.Linear_emo_graph(GCN_feat_batch)
        # GCN_feat_batch = torch.add(GCN_feat_batch_ori, GCN_feat_batch)
        # GCN_feat_batch = self.share_linear_block(GCN_feat_batch)
        b, c, hw = GCN_feat_batch.shape
        h = w = int(hw ** 0.5) 
        GCN_feat_batch = GCN_feat_batch.view(b, c, h, w)
        # GCN_feat_batch = GCN_feat_batch.view(b, c h, w)
        # import pdb;pdb.set_trace()
        GCN_feat_batch = self.conv_emb(GCN_feat_batch)
        return GCN_feat_batch # [b, 512, 32, 32] # [B,512,32,32] b c h w -> b h w , c


class EmbeddingGroup(nn.Module):
    def __init__(self, device='cpu', dtype=torch.float32):
        super(EmbeddingGroup, self).__init__()  # 确保正确调用 nn.Module 的初始化
        self.device = device
        self.beta = 0.25
        self.embedding_num = 8
        self.emotion_keys = [
            'amusement', 'anger', 'awe', 'contentment',
            'disgust', 'excitement', 'fear', 'sadness'
        ]
        self.embedding_group = nn.ModuleDict()  # 使用 nn.ModuleDict 代替字典
        self.n_e = 256
        self.e_dim = 1024
        self.dtype = dtype
        
        self.VAD_dict = ['V','A','D']
        # emotion_wheel = ['amusement', 'contentment', 'awe', 'excitement', 'fear', 'sadness', 'disgust', 'anger']
        # 定义可学习的均值和方差
        self.means = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(3, dtype=dtype, device=device))
            for key in self.emotion_keys
        })
        # self.means = nn.ParameterDict({
        #     key: nn.Parameter(torch.eye(8, dtype=dtype, device=device)[i] * 3.0)
        #     for i, key in enumerate(self.emotion_keys)
        # })
        # self.means = self.initialize_means(emotion_wheel, dim=8)
        self.stds = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))
            for key in self.emotion_keys
        })

        for param in self.means.parameters():
            param.requires_grad = True

        for param in self.stds.parameters():
            param.requires_grad = True

        # 初始化 embedding
        for key in self.emotion_keys:
            embedding = nn.Embedding(self.n_e, self.e_dim).to(self.dtype)
            # 使用 learnable 均值和方差初始化
            with torch.no_grad():
                # embedding.weight.data.normal_(mean=torch.sum(self.means[key]), std=torch.sum(self.stds[key]))
                # mean_value = torch.mean(self.means[key]).item()  # 转换为标量
                # std_value = self.stds[key].item()    # 转换为标量
                # embedding.weight.data.normal_(mean=mean_value, std=std_value)
                embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
                self.embedding_group[key] = embedding
                
                

        # 定义卷积层
        self.conv_z = nn.Conv2d(kernel_size=(1, 1), stride=1, in_channels=512, out_channels=512).to(self.device).to(self.dtype)

        '''
        amusement	0.929	0.837	0.803
        anger	0.167	0.865	0.657
        awe	        0.469	0.74	0.3
        contentment	0.875	0.61	0.782
        disgust	0.052	0.775	0.317
        excitement	0.896	0.684	0.731
        fear   0.073	0.84	0.293
        sadness	0.052	0.288	0.164
        '''
        scale = 10.0
        self.target_means = {
            "amusement":((torch.tensor([0.929,0.837,0.803]) * scale) - 5.0).to(self.device).to(self.dtype),
            "anger":((torch.tensor([0.167, 0.865, 0.657]) * scale) - 5.0).to(self.device).to(self.dtype),
            "awe":((torch.tensor([0.469, 0.74, 0.3]) * scale) - 5.0).to(self.device).to(self.dtype),
            "contentment":((torch.tensor([0.875, 0.61, 0.782]) * scale)- 5.0).to(self.device).to(self.dtype),
            "disgust":((torch.tensor([0.052, 0.775, 0.317]) * scale)- 5.0).to(self.device).to(self.dtype),
            "excitement":((torch.tensor([0.896, 0.684, 0.731]) * scale)- 5.0).to(self.device).to(self.dtype),
            "fear":((torch.tensor([0.073, 0.84, 0.293]) * scale)- 5.0).to(self.device).to(self.dtype),
            "sadness":((torch.tensor([0.052, 0.288, 0.164]) * scale)- 5.0).to(self.device).to(self.dtype),
            }
        # import pdb;pdb.set_trace()
        # print(self.target_means)
        # 禁止梯度计算
        for key in self.target_means:
            self.target_means[key].requires_grad = False

    # all_mean
    def mean_difference_loss(self, lambda_diff=1.0):
        """
        增加所有情感向量均值的差异性
        means: nn.ParameterDict，存储所有情感向量的均值
        lambda_diff: 控制损失强度的权重
        """
        # 将所有情感均值堆叠成一个张量
        means_tensor = torch.stack([self.means[key] for key in self.means.keys()])
        
        # 计算两两向量之间的欧几里得距离
        pairwise_diff = torch.cdist(means_tensor, means_tensor, p=2)  # [N, N]
        
        # 只取上三角矩阵的非对角线部分，防止重复计算
        loss = -torch.sum(pairwise_diff.triu(diagonal=1))  # 负号表示最大化差异
        
        return lambda_diff * loss

    # a mean
    def intra_vector_difference_loss(self, key, lambda_intra=1.0):
        """
        增加单个情感向量内部元素的差异性
        mean: 单个情感向量（torch.Tensor）
        lambda_intra: 控制损失强度的权重
        """
        # 计算两两元素之间的差值矩阵
        mean = self.means[key]
        diff_matrix = mean.unsqueeze(0) - mean.unsqueeze(1)  # [3, 3]
        
        # 计算差值的绝对值，并最小化其和
        loss = -torch.sum(torch.abs(diff_matrix.triu(diagonal=1)))  # 负号表示最大化差异
        
        return lambda_intra * loss
    def emotion_similarity_loss(self, key,  lambda_similarity = 0.5):
        means_tensor = torch.stack([self.means[key] for key in self.emotion_keys])
        mean_diff_loss = -torch.mean((means_tensor.unsqueeze(0) - means_tensor.unsqueeze(1)).pow(2))  # 增加均值的差异性
        return mean_diff_loss * lambda_similarity
    
    def emotion_direction_loss(self, key, lambda_direction=0.5):
        target_mean = self.target_means[key]
        current_mean = self.means[key]
        mean_direction_loss = torch.mean((current_mean - target_mean) ** 2)
        return lambda_direction * mean_direction_loss

    def emotion_regularization_std_loss(self, key, lambda_regularization=0.5):
        target_std = 1.0
        std_regularization_loss = torch.mean((self.stds[key] - target_std) ** 2)
        return std_regularization_loss * lambda_regularization

    # def orthogonal_regularization_loss(self, key, lambda_ortho=0.1):
    #     means_tensor = torch.stack([self.means[key] for key in self.emotion_keys])
    #     ortho_loss = torch.norm(means_tensor @ means_tensor.T - torch.eye(len(self.emotion_keys), device=self.device))
    #     return lambda_ortho * ortho_loss


    def forward(self, z, key):
        # 使用当前的means和stds调整embedding
        # with torch.no_grad():
        std = abs(self.stds[key]).item() + torch.randn_like(self.stds[key]).item()
        noise_std = std
        mean = torch.mean(self.means[key]).item() # + torch.mean(torch.randn_like(self.means[key])) * noise_std 
        
        # 获取对应的 embedding
        embedding = self.embedding_group[key].to(self.device).to(self.dtype)
        
        # 对 embedding 权重进行动态标准化（不冻结它们）
        embedding_weight = embedding.weight * std + mean  # 使用当前的 mean 和 std

        # 前向传播的距离计算使用 embedding_weight 而非固定的 embedding.weight
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, self.e_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, embedding_weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device, dtype=z.dtype)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, embedding_weight).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()

        # emotion_similarity_loss = self.emotion_similarity_loss(key, 1)
        # direction_loss = self.emotion_direction_loss(key, 1)
        # mean_difference_loss = self.mean_difference_loss(0.3)
        # intra_vector_difference_loss = self.intra_vector_difference_loss(key, 1)
        # regularization_loss = self.emotion_regularization_std_loss(key,1)

        # total_loss = loss + intra_vector_difference_loss +  emotion_similarity_loss  + direction_loss + regularization_loss 

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # import pdb;pdb.set_trace()

        # z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q = self.conv_z(z_q)

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


class MLP(nn.Module): 
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, num_i, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_o)

    def forward(self, x):
        x = self.linear1(x)
        return x


# Discriminator Network
class Discriminator(torch.nn.Module): 
    def __init__(self, device, dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # self.bce_loss = nn.BCELoss()
                     

        self.fc_block_1 = torch.nn.Sequential(*[
            nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=512,out_channels=512),
            nn.LeakyReLU(0.2, True)]
            ).to(self.device)

        self.fc_block_2 = torch.nn.Sequential(*[
            nn.Linear(512*16*16, 512), 
            # nn.BatchNorm1d(1024), 
            # nn.LayerNorm(1024),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(1024, 512),
            nn.BatchNorm1d(512), 
            # nn.LayerNorm(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 8),
            nn.Sigmoid()
            ]).to(self.device)
            # nn.BatchNorm1d(512), 
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(512, 1)]).to(self.device)

        for param in self.fc_block_1.parameters():
            param.requires_grad = True
        for param in self.fc_block_2.parameters():
            param.requires_grad = True

        self.softmax = F.softmax


    def forward(self, feats):
        # import pdb;pdb.set_trace()
        b,c,h,w = feats.shape
        # feats = feats.view(b,c*h*w).to(self.device)
        out1 = self.fc_block_1(feats).to(self.device)
        out1 = out1.view(b,c*h*w).to(self.device)
        out = self.fc_block_2(out1).to(self.device)
        out = self.softmax(out, dim=-1)
        return out # [b, 8]

class EmotionTokenModule(nn.Module):
    def __init__(self, do, da, dr, ds, dg, in_ch, hidden_ch, out_ch, device='cpu'):
        super(EmotionTokenModule, self).__init__()
        self.device = device
        self.EmbeddingGroup = EmbeddingGroup(self.device)
        self.clip_text_encoder = ClipTextEncoder(self.device)
        self.emotion_forest_encoder = GraphEncoder(do, da, dr, ds, dg, self.device)
        self.emo_keys = [
            'amusement', 'anger', 'awe', 'contentment',
            'disgust', 'fear', 'excitement', 'sadness'
        ]
        for param in self.EmbeddingGroup.parameters():
            param.requires_grad = True
        for param in self.emotion_forest_encoder.parameters():
            param.requires_grad = True


    def forward(self, data):
        _, _, sentiment_word_batch, _ = data 
        emo_text = [item[0] for item in sentiment_word_batch]
        graph_feats = self.emotion_forest_encoder(data)
        embedding_output_list = []
        embedding_output_list_inkey = []
        loss_list = []
        perplexity_list = []
        min_encodings_list = []
        min_encoding_indices_list = []

        emo_text_temp = []

        # 8个码本都训练
        for index, word in enumerate(emo_text):
            embedding_output, loss, (perplexity, min_encodings, min_encoding_indices) = self.EmbeddingGroup(graph_feats[index].unsqueeze(dim=0), word)
            embedding_output_list_inkey.append(embedding_output)
            loss_list.append(loss)
            perplexity_list.append(perplexity)
            min_encodings_list.append(min_encodings)
            min_encoding_indices_list.append(min_encoding_indices)

        embedding_output_list_inkey = torch.cat(embedding_output_list_inkey, dim=0)
        return embedding_output_list_inkey, loss_list, perplexity_list, min_encodings_list, min_encoding_indices_list

def test():
    do = 512  # Object embedding dimension from CLIP 对象
    da = 512  # Adjective embedding dimension from CLIP 形容词
    ds = 512   # Relationship embedding dimension 关系
    dr = 512   # Relationship embedding dimension 关系
    dg = 512  # Global embedding dimension 全局词
    in_ch = 512
    hidden_ch = 512
    out_ch = 512
    images = torch.randn(2, 3, 256, 256)  # Batch of 16 images

    object_words_batch = [['sky', 'flower', 'mountain','cat', 'dog', 'tree']]
    adjective_words_batch = [["colorful", 'beautiful', 'beautiful',"small", "cute", "tall"]]
    sentiment_word_batch = [["awe"]]
    global_adj_word_batch = [["colorful"]]

    data = (object_words_batch, adjective_words_batch, sentiment_word_batch, global_adj_word_batch) # obsg
    device = torch.device('cuda:2')

    model = EmotionTokenModule(do, da, dr, ds, dg, in_ch, hidden_ch, out_ch, device)

    encoder = ImageEncoder(device=device)
    img = torch.rand(2,3,256,256)
    out_img = encoder(img)

    out, loss, _, _, _, _ = model(data)
    print(out.shape)
    print(loss)

if __name__ == "__main__":
    test()