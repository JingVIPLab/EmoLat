import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
# from emo_token_gen_model.emo_token_gen_model import EmbeddingGroup as EmbeddingGroup

# 假设你已经有 `EmbeddingGroup` 类定义在这
# 从模型中提取嵌入并准备数据进行 t-SNE

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
        print(self.target_means)
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

def extract_embeddings(model, keys, num_samples=1000):
    embeddings = []
    labels = []
    
    # 从模型中提取嵌入特征
    for key in keys:
        embedding = model[key].weight.data.cpu().numpy()
        sampled_embeddings = embedding[:num_samples]
        embeddings.append(sampled_embeddings)
        labels += [key] * len(sampled_embeddings)

    embeddings = np.vstack(embeddings)
    return embeddings, labels


# 修改后的可视化函数，使用不同颜色区分不同的情感
def visualize_embeddings(embeddings_2d, labels, keys, save_path='tsne_visualization.png'):
    plt.figure(figsize=(10, 8))
    
    # 使用 colormap 为每个情感分配不同的颜色
    colors = cm.get_cmap('tab10', len(keys))  # 使用 'tab10' colormap（最多10种颜色）
    
    for idx, key in enumerate(keys):
        indices = [i for i, label in enumerate(labels) if label == key]
        plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1], 
            label=key, 
            color=colors(idx)  # 根据情感类别分配颜色
        )
    
    plt.legend()
    plt.title('t-SNE visualization of emotion embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # 保存图像
    plt.savefig(save_path, format='png', dpi=300)  # 保存为高质量图片，dpi 可以根据需要调整
    plt.show()


def load_model(model_struct, model_name, device):
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
    return model_struct

# 初始化你的模型
id = 0
device = torch.device(f"cuda:{id}")
emb_model = EmbeddingGroup(device=device)
# model = torch.load(f"/home/zjx/lab/my3_lab/v1/checkpoint/embedding_group_final.pth")

# 读取模型
emb_group = load_model(emb_model.embedding_group, "/home/zjx/lab/my3_lab/v1/checkpoint/embedding_group_final.pth", device)
emb_means = load_model(emb_model.means, "/home/zjx/lab/my3_lab/v1/checkpoint/mean_final.pth", device)
emb_stds = load_model(emb_model.stds, "/home/zjx/lab/my3_lab/v1/checkpoint/std_final.pth", device)

emotion_keys = [
            'amusement', 'anger', 'awe', 'contentment',
            'disgust','excitement', 'fear', 'sadness'
        ]

# 提取所有情感类别的嵌入矩阵并组合
all_embeddings = []
all_labels = []

for emotion_key in emotion_keys:
    # 提取情感类别的嵌入
    embedding_weights = emb_group[emotion_key].weight.detach().cpu().numpy()
    # 提取对应的均值和标准差
    # mean_value = torch.mean(emb_means[emotion_key]).item()
    # std_value = emb_stds[emotion_key].item()
    # print(mean_value)
    # print(std_value)
    # print()
    
    # import pdb;pdb.set_trace()
    mean_value = embedding_weights.mean(axis=1)[0]
    std_value = np.var(embedding_weights, axis=0)
    
    # 对嵌入进行标准化 (减去均值然后除以标准差)
    standardized_embedding_weights = (embedding_weights - mean_value)  / (std_value + 1e-10)
    print(mean_value)
    print(std_value)
    print()
    
    # 保存标准化后的嵌入
    all_embeddings.append(standardized_embedding_weights)
    all_labels += [emotion_key] * standardized_embedding_weights.shape[0]

# 合并所有情感类别的嵌入
all_embeddings = np.vstack(all_embeddings)

from sklearn.manifold import TSNE

# 使用 t-SNE 进行降维，将高维数据降到 2 维空间
tsne = TSNE(n_components=2, perplexity=35,n_iter=600,random_state=47)
reduced_embeddings = tsne.fit_transform(all_embeddings)  # 形状: (所有嵌入的数量, 2)


import matplotlib.pyplot as plt

# 绘制 t-SNE 降维后的嵌入
plt.figure(figsize=(10, 10))

# 为了区分不同情感类别，使用不同的颜色
colors = plt.cm.get_cmap('tab10', len(emotion_keys))
for i, emotion_key in enumerate(emotion_keys):
    indices = [index for index, label in enumerate(all_labels) if label == emotion_key]
    plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=emotion_key, alpha=0.5, c=[colors(i)])

plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

save_path = "/home/zjx/lab/my3_lab/v1/vis_img/tsne_visualization_1.png"
plt.legend()
plt.title('Visualization of emotion space')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
plt.savefig(save_path, format='png', dpi=300)
plt.show()
