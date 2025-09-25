import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import torch.optim as optim
from torch_geometric.nn import GCNConv
from collections import OrderedDict
from tqdm import tqdm
from data.emosetplus import EmoAdjDataset
from emo_token_gen_model.emo_token_gen_model import EmotionTokenModule, ImageEncoder, Discriminator
from torch.utils.data import DataLoader
import os
from model.vgg import vgg

BCE_Loss = nn.BCELoss()
MSE_Loss = nn.MSELoss()
adversarial_loss = nn.CrossEntropyLoss()

emotion_dict = {
    "amusement":torch.tensor([1.,0,0,0,0,0,0,0]),
    "anger":torch.tensor([0,1.,0,0,0,0,0,0]),
    "awe":torch.tensor([0,0,1.,0,0,0,0,0]),
    "contentment":torch.tensor([0,0,0,1.,0,0,0,0]),
    "disgust":torch.tensor([0,0,0,0,1.,0,0,0]),
    "excitement":torch.tensor([0,0,0,0,0,1.,0,0]),
    "fear":torch.tensor([0,0,0,0,0,0,1.,0]),
    "sadness":torch.tensor([0,0,0,0,0,0,0,1.]),
    }

emotion_dict_fake = {
    "amusement":torch.tensor([0,1.,1.,1.,1.,1.,1.,1.]),
    "anger":torch.tensor([1.,0,1.,1.,1.,1.,1.,1.]),
    "awe":torch.tensor([1.,1.,0,1.,1.,1.,1.,1.]),
    "contentment":torch.tensor([1.,1.,1.,0,1.,1.,1.,1.]),
    "disgust":torch.tensor([1.,1.,1.,1.,0,1.,1.,1.]),
    "excitement":torch.tensor([1.,1.,1.,1.,1.,0,1.,1.]),
    "fear":torch.tensor([1.,1.,1.,1.,1.,1.,0,1.]),
    "sadness":torch.tensor([1.,1.,1.,1.,1.,1.,1.,0]),
    }


# 获取数据集
def get_dataset(batch_size=1,proportion=1):
    image_root = '/home/public/datasets/EmoSet-118K/image'
    json_root = '/home/zjx/lab/dataset/emoset_v1'
    dataset = EmoAdjDataset(img_root=image_root,
                            json_root=json_root,
                            proportion=proportion,
                            is_train=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    return dataloader

def pair_filter(pair_batch):
    new_pair_batch = []
    for pair in pair_batch:
        new_pair = []
        for p in pair:
            if p == ['<pad>', '<pad>']:
                continue
            else:
                new_pair.append(p)
        new_pair_batch.append(new_pair)
    return new_pair_batch

# 获得参数
def get_model_param():
    do = 512  # Object embedding dimension from CLIP 对象
    da = 512  # Adjective embedding dimension from CLIP 形容词
    ds = 512   # Relationship embedding dimension 关系
    dr = 512   # Relationship embedding dimension 关系
    dg = 512  # Global embedding dimension 全局词
    in_ch = 512 
    hidden_ch = 512
    out_ch = 512
    return do, da, dr, ds, dg, in_ch, hidden_ch, out_ch

# 获得模型
def creat_model(id):
    device = torch.device(f'cuda:{id}')
    do, da, dr, ds, dg, in_ch, hidden_ch, out_ch = get_model_param()
    model = EmotionTokenModule(do, da, dr, ds, dg, in_ch, hidden_ch, out_ch, device)
    imageEncoder = ImageEncoder(vgg, device)
    discriminator = Discriminator(device)
    return model, imageEncoder, discriminator

def change_label_form(emo,adj,pair):
    object_words_batch = []
    adjective_words_batch = []
    sentiment_word_batch = []
    global_adj_word_batch = []
    for i in range(len(emo)):
        # import pdb;pdb.set_trace()
        obj_batch = []
        adj_batch = []
        for j in range(len(pair[i])):
            obj_batch.append(pair[i][j][0])
            adj_batch.append(pair[i][j][1])
        object_words_batch.append(obj_batch)
        adjective_words_batch.append(adj_batch)
        sentiment_word_batch.append(emo[i])
        global_adj_word_batch.append(adj[i])
        
    
    return object_words_batch, adjective_words_batch, sentiment_word_batch, global_adj_word_batch 

def cal_loss_total(modules, optimizer, images, data, num_batches, G_D_step, device):
    # get optimzer and modules
    loss_aug = 10
    _, _, sentiment_word_batch, _ = data 

    optimizer_G,  optimizer_D = optimizer
    imageEncoder, graphModel, discriminator = modules
    
    Tensor = torch.FloatTensor
    
    # get label
    with torch.no_grad():
        real_data = imageEncoder(images)
    b, _, _, _ = real_data.shape
    # real_label = Tensor(b,1).fill_(0.9).to(device).detach()
    # fake_label = Tensor(b,1).fill_(0.1).to(device).detach()

    # Train Discrinamtor
    graph_feats,  loss_list, perplexity_list, min_encodings_list, min_encoding_indices_list = graphModel(data)
    embeding_group_loss = sum(loss_list)
    optimizer_D.zero_grad()

    # import pdb;pdb.set_trace()
    emo_text = [item[0] for item in sentiment_word_batch]
    real_label = []
    fake_label = []
    for index, text in enumerate(emo_text):
        real_label.append(emotion_dict[text].unsqueeze(0))
    real_label = torch.cat(real_label, dim=0).to(device).detach()
    for index, text in enumerate(emo_text):
        fake_label.append(emotion_dict_fake[text].unsqueeze(0))
    fake_label = torch.cat(fake_label, dim=0).to(device).detach()

    real_loss = adversarial_loss(discriminator(real_data).to(device), real_label.to(device))
    fake_loss = adversarial_loss(discriminator(graph_feats).to(device).detach(), fake_label.to(device))

    d_loss_D = (real_loss + fake_loss) / 2 
    d_loss_D *= loss_aug
    d_loss_D.backward()
    optimizer_D.step()

    # Train GraphEncoder_Generator
    if num_batches % G_D_step == 0:
        optimizer_G.zero_grad()
        out_feats_graph = discriminator(graph_feats)
        g_loss_G = adversarial_loss(out_feats_graph.to(device), real_label.to(device)) + embeding_group_loss
        g_loss_G *= loss_aug
        g_loss_G.backward()        
        optimizer_G.step()

    # print("means", graphModel.EmbeddingGroup.means.items())
    print("mean")
    for key, value in graphModel.EmbeddingGroup.means.items():
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            print(torch.mean(value.detach()).item(), end=' ')
    print()    
    print("sum")
    for key, value in graphModel.EmbeddingGroup.means.items():
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            print(torch.sum(value.detach()).item(), end=' ')
    print()    
    modules = (imageEncoder, graphModel, discriminator)

    if num_batches % G_D_step == 0:
        g_loss = g_loss_G
        d_loss = d_loss_D
    else:
        g_loss = None
        d_loss = d_loss_D
    return g_loss, d_loss, modules

def train_emotion_token_module(
    model,
    dataset_list,
    device,
    save_dir,
    batch_size=1,
    num_epochs=4,
    learning_rate=1e-6,
):

    graphModel, imageEncoder, discriminator = model
    

    # optimizer_G = optim.AdamW(
    #     list(graphModel.emotion_forest_encoder.parameters()) + 
    #     list(graphModel.EmbeddingGroup.embedding_group.parameters()) + 
    #     list(graphModel.EmbeddingGroup.means.values()) + 
    #     list(graphModel.EmbeddingGroup.stds.values()), 
    #     lr=learning_rate,
    #     betas=(0.9, 0.98)
    # )
    optimizer_G = optim.AdamW([
        {'params': graphModel.emotion_forest_encoder.parameters(), 'lr': learning_rate, 'betas': (0.9, 0.98)},
        {'params': graphModel.EmbeddingGroup.embedding_group.parameters(), 'lr': learning_rate, 'betas': (0.9, 0.98)},
        {'params': graphModel.EmbeddingGroup.means.values(), 'lr': learning_rate*10 , 'betas': (0.9, 0.98), 'weight_decay': 1e-4},
        {'params': graphModel.EmbeddingGroup.stds.values(), 'lr': learning_rate*10, 'betas': (0.9, 0.98), 'weight_decay': 1e-4}
    ])
    # optimizer_G_E = optim.AdamW(graphModel.EmbeddingGroup.parameters(), lr=learning_rate,betas=(0.9, 0.98))
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=learning_rate , betas=(0.9, 0.98))

    optimizers = (optimizer_G, optimizer_D)

    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=4, gamma=0.9)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=4, gamma=0.9)


    graphModel.train()  
    discriminator.train()
    imageEncoder.eval()

    modules = (imageEncoder, graphModel, discriminator)
    # 创建一个外层的 tqdm 进度条用于 epoch
    epoch_progress = tqdm(range(0, num_epochs), desc="Epochs", unit="epoch")
    torch.autograd.set_detect_anomaly(True)
    G_D_step = 5
    for epoch in epoch_progress:

        num_batches = 0
        g_loss_G_mean_max = float('inf')
        d_loss_D_mean_max = float('inf')
        g_loss_G_sum = 0
        d_loss_D_sum = 0
        for index, dataset in enumerate(dataset_list):
    
            batch_progress = tqdm(dataset, desc=f"{index} Epoch {epoch}/{num_epochs}", leave=False, unit="batch")

            for json_path, images, emo, adj, pair in batch_progress:
                
                g_loss_G_sum_batch = 0
                d_loss_D_sum_batch = 0
                
                batch, _, _, _ = images.shape
                emo = [[e] for e in emo]
                adj = [[a] for a in adj]
                pair_list = []
                try:
                    for b in range(batch_size):
                        formatted_data = [[item[b] for item in p] for p in pair]
                        pair_list.append(formatted_data)
                except Exception as e:
                    continue
                pair = pair_list
                pair = pair_filter(pair_list)
                images = images.to(device)
                object_words_batch, adjective_words_batch, sentiment_word_batch, global_adj_word_batch = change_label_form(emo,adj,pair)

                data = (object_words_batch, adjective_words_batch, sentiment_word_batch, global_adj_word_batch)
                
                g_loss, d_loss, modules = cal_loss_total(modules, optimizers, images, data, num_batches, G_D_step, device)

                g_loss_G = g_loss
                d_loss_D = d_loss  

                # Accumulate generator and discriminator loss
                d_loss_D_sum += d_loss_D.item()
                if num_batches % G_D_step == 0:
                    g_loss_G_sum += g_loss_G.item()
                
                # Calculate mean generator loss for the epoch
                d_loss_D_sum_batch += d_loss_D.item()
                if num_batches % G_D_step == 0:
                    g_loss_G_sum_batch += g_loss_G.item()

                d_loss_D_mean_batch = d_loss_D_sum_batch / batch if batch > 0 else 0.0
                if num_batches % G_D_step == 0:
                    g_loss_G_mean_batch = g_loss_G_sum_batch / batch if batch > 0 else 0.0
                    # print(f"Epoch [{epoch}], g_loss_G_1_mean_batch: {g_loss_G_1_mean_batch:.4f}, g_loss_G_2_mean_batch: {g_loss_G_2_mean_batch:.4f}")

                # print(f"Epoch [{epoch}], d_loss_D_1_mean_batch: {d_loss_D_1_mean_batch:.4f}, d_loss_D_2_mean_batch: {d_loss_D_2_mean_batch:.4f}")
                num_batches += 1
                
            scheduler_D.step()
            scheduler_G.step()

        # Calculate mean generator loss for the epoch
        d_loss_D_mean = d_loss_D_sum / num_batches if num_batches > 0 else 0.0
        g_loss_G_mean = g_loss_G_sum / num_batches if num_batches > 0 else 0.0
        print(f"Epoch [{epoch}/{num_epochs}], d_loss_D_mean: {d_loss_D_mean:.4f}")
        print(f"Epoch [{epoch}/{num_epochs}], g_loss_G_mean: {g_loss_G_mean:.4f}")

        # Step the learning rate scheduler
        if (epoch + 1) % 1 == 0 and g_loss_G_mean < g_loss_G_mean_max:
            e_loss_G_mean_max = g_loss_G_mean
            e_final_save_path = os.path.join(save_dir,f'embedding_group_final.pth')
            torch.save(graphModel.EmbeddingGroup.embedding_group.state_dict(), e_final_save_path)

            mean_final_save_path = os.path.join(save_dir,f'mean_final.pth')
            print(graphModel.EmbeddingGroup.means.items())
            torch.save(graphModel.EmbeddingGroup.means.state_dict(), mean_final_save_path)

            std_final_save_path = os.path.join(save_dir,f'std_final.pth')
            print(graphModel.EmbeddingGroup.stds.items())
            torch.save(graphModel.EmbeddingGroup.stds.state_dict(), std_final_save_path)

            g_final_save_path = os.path.join(save_dir,f'graph_encoder_final.pth')
            torch.save(graphModel.emotion_forest_encoder.state_dict(), g_final_save_path)

        if (epoch + 1) % 1 == 0 and d_loss_D_mean < d_loss_D_mean_max:
            d_loss_D_mean_max = d_loss_D_mean
            d_final_save_path = os.path.join(save_dir,f'discriminator_final.pth')
            torch.save(discriminator.state_dict(), d_final_save_path)
        
    


# 训练函数
def train():
    gpu_id = 3
    batch_size = 64
    num_epochs = 10
    learning_rate = 5e-4
    proportion = 0.2
    save_dir = "/home/zjx/lab/my3_lab/v1/checkpoint"
    dataloader_list = []
    steps = int(1/proportion)
    p = proportion
    for i in range(steps):
        print(i)
        print(p)    
        dataloader = get_dataset(batch_size, p)  # 获取数据集
        dataloader_list.append(dataloader)
        p = round(p + proportion, 1) 
    graphModel, imageEncoder, discriminator = creat_model(gpu_id)
    model = graphModel, imageEncoder, discriminator
    device = torch.device(f"cuda:{gpu_id}")
    train_emotion_token_module(model, dataloader_list,device,save_dir,learning_rate=learning_rate,batch_size=batch_size, num_epochs=num_epochs)

if __name__ == "__main__":
    train()
