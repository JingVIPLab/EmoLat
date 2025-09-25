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
        
        # 定义可学习的均值和方差
        self.means = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device))
            for key in self.emotion_keys
        })
        self.stds = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))
            for key in self.emotion_keys
        })

        # 初始化 embedding
        for key in self.emotion_keys:
            embedding = nn.Embedding(self.n_e, self.e_dim).to(self.dtype)
            self.embedding_group[key] = embedding
            # 使用 learnable 均值和方差初始化
            with torch.no_grad():
                embedding.weight.data.normal_(mean=self.means[key].item(), std=self.stds[key].item())
        
        # 定义卷积层
        self.conv_z = nn.Conv2d(kernel_size=(1, 1), stride=1, in_channels=512, out_channels=512).to(self.device).to(self.dtype)

    def forward(self, z, key):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # 根据提供的 key 选择对应的 embedding
        embedding = self.embedding_group[key].to(self.device).to(self.dtype)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device, dtype=z.dtype)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, embedding.weight).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        
        e_mean = torch.mean(min_encodings, dim=0)

        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

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
            'disgust', 'excitement', 'fear','sadness'
        ]

# 提取嵌入
embeddings, labels = extract_embeddings(emb_group, emotion_keys)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, perplexity=35, n_iter=900, random_state=66)
embeddings_2d = tsne.fit_transform(embeddings)
# 可视化并保存图片
save_path = "/home/zjx/lab/my3_lab/v1/vis_img/tsne_visualization.png"

visualize_embeddings(embeddings_2d, labels, emotion_keys, save_path=save_path)