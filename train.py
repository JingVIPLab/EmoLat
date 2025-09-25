import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import torch.optim as optim
from torch_geometric.nn import GCNConv
from collections import OrderedDict
from tqdm import tqdm
from data.emosetplus import EmoAdjDataset
from emo_token_gen_model.emo_token_gen_model import EmotionTokenModule
from torch.utils.data import DataLoader
import os

BCE_Loss = nn.BCELoss()
MSE_Loss = nn.MSELoss()
adversarial_loss = torch.nn.MSELoss()


# 获取数据集
def get_dataset(batch_size=1,proportion=1):
    image_root = '/home/public/datasets/EmoSet-118K/image'
    json_root = '/home/zjx/lab/dataset/emoset_v1'
    dataset = EmoAdjDataset(img_root=image_root,
                            json_root=json_root,
                            proportion=proportion,
                            is_train=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
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
    return model()

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

def cal_loss_total(modules, optimizer, images, data, num_batches,  G_D_step, device):
    # get optimzer and modules
    loss_aug = 10
    _, _, sentiment_word_batch, _ = data 
    optimizer_G_1, optimizer_G_2, optimizer_G_3, optimizer_1, optimizer_2, optimizer_3  = optimizer
    ImageEncoder, GraphEncoder, TextEncoder, Discriminator, alpha_logit = modules

    Discriminator_1, Discriminator_2, Discriminator_3 = Discriminator
    
    Tensor = torch.FloatTensor
    
    # get label
    with torch.no_grad():
        real_data = ImageEncoder(images)
    b, _ = real_data.shape
    real_label = Tensor(b,1).fill_(0.9).to(device).detach()
    fake_label = Tensor(b,1).fill_(0.1).to(device).detach()

    ###############################################
    # train discriminator_1 and Graph_encoder
    ###############################################
    
    # teacher and good student

    # Train Discrinamtor1
    optimizer_1.zero_grad()
    graph_feats = GraphEncoder(data)
    real_loss = adversarial_loss(Discriminator_1(real_data), real_label)
    fake_loss = adversarial_loss(Discriminator_1(graph_feats).detach(), fake_label)
    d_loss_D_1 = (real_loss + fake_loss) / 2
    d_loss_D_1 *= loss_aug
    d_loss_D_1.backward()
    optimizer_1.step()

    if num_batches % G_D_step == 0:
    # Train GraphEncoder_Generator
        optimizer_G_1.zero_grad()
        out_feats_graph = Discriminator_1(graph_feats)
        g_loss_G_1 = adversarial_loss(out_feats_graph.to(device), real_label.to(device))
        g_loss_G_1 *= loss_aug
        g_loss_G_1.backward()
        optimizer_G_1.step()


    ##############################################
    # train discriminator_2
    #####################################
    # teacher and normal student

    # Train Discriminator_2
    optimizer_2.zero_grad()
    text_feats = TextEncoder(sentiment_word_batch)
    real_loss = adversarial_loss(Discriminator_2(real_data), real_label)
    fake_loss = adversarial_loss(Discriminator_2(text_feats).detach(), fake_label)
    d_loss_D_2 = (real_loss + fake_loss) / 2
    d_loss_D_2 *= loss_aug
    d_loss_D_2.backward()
    optimizer_2.step()

    if num_batches % G_D_step == 0:
        # Train TextEncoder
        optimizer_G_2.zero_grad()
        out_feats_text = Discriminator_2(text_feats)
        g_loss_G_2 = adversarial_loss(out_feats_text.to(device), real_label.to(device))
        g_loss_G_2 *= loss_aug
        g_loss_G_2.backward()
        optimizer_G_2.step()

    ##############################################
    # train discriminator_3
    #####################################
    # good and normal student

    # Train Discriminator_3
    optimizer_3.zero_grad()
    with torch.no_grad():
        real_data = GraphEncoder(data).detach()
    text_feats = TextEncoder(sentiment_word_batch)
    real_loss = adversarial_loss(Discriminator_3(real_data), real_label)
    fake_loss = adversarial_loss(Discriminator_3(text_feats).detach(), fake_label)
    d_loss_D_3 = (real_loss + fake_loss) / 2
    d_loss_D_3 *= loss_aug
    d_loss_D_3.backward()
    optimizer_3.step()

    if num_batches % G_D_step == 0:
        # Train TextEncoder
        optimizer_G_3.zero_grad()
        out_feats_text = Discriminator_3(text_feats)
        g_loss_G_3 = adversarial_loss(out_feats_text.to(device), real_label.to(device))
        g_loss_G_3 *= loss_aug
        g_loss_G_3.backward()
        optimizer_G_3.step()



    modules = (ImageEncoder, GraphEncoder, TextEncoder, Discriminator, alpha_logit)

    if num_batches % G_D_step == 0:
        g_loss = (g_loss_G_1, g_loss_G_2, g_loss_G_3)
        d_loss = (d_loss_D_1, d_loss_D_2, d_loss_D_3)
    else:
        g_loss = (None,None)
        d_loss = (d_loss_D_1, d_loss_D_2, d_loss_D_3)
    return g_loss, d_loss, modules

def train_emotion_token_module(
    model,
    dataset,
    device,
    save_dir,
    batch_size=1,
    num_epochs=4,
    learning_rate=1e-6,
):

    ImageEncoder, GraphEncoder, TextEncoder, Discriminator_1, Discriminator_2, Discriminator_3, alpha_logit = model

    optimizer_G_1 = optim.AdamW(GraphEncoder.parameters(), lr=learning_rate,betas=(0.9, 0.98))
    optimizer_G_2 = optim.AdamW(TextEncoder.parameters(), lr=learning_rate * 10,betas=(0.9, 0.98))
    optimizer_G_3 = optim.AdamW(TextEncoder.parameters(), lr=learning_rate,betas=(0.9, 0.98))

    optimizer_1 = optim.AdamW(Discriminator_1.parameters(), lr=learning_rate * 10, betas=(0.9, 0.98))
    optimizer_2 = optim.AdamW(Discriminator_2.parameters(), lr=learning_rate * 10, betas=(0.9, 0.98))
    optimizer_3 = optim.AdamW(Discriminator_3.parameters(), lr=learning_rate, betas=(0.9, 0.98))

    optimizers = (optimizer_G_1, optimizer_G_2, optimizer_G_3, optimizer_1, optimizer_2, optimizer_3)

    scheduler_G_1 = optim.lr_scheduler.StepLR(optimizer_G_1, step_size=4, gamma=0.9)
    scheduler_G_2 = optim.lr_scheduler.StepLR(optimizer_G_2, step_size=4, gamma=0.9)
    scheduler_G_3 = optim.lr_scheduler.StepLR(optimizer_G_3, step_size=4, gamma=0.9)
    scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=4, gamma=0.9)
    scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, step_size=4, gamma=0.9)
    scheduler_3 = optim.lr_scheduler.StepLR(optimizer_3, step_size=4, gamma=0.9)
    # scheduler = (scheduler_G_1, scheduler_G_2, scheduler_D)

    Discriminator = (Discriminator_1, Discriminator_2, Discriminator_3)

    modules = (ImageEncoder, GraphEncoder,TextEncoder,Discriminator, alpha_logit)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # model.train()
    GraphEncoder.train() 
    TextEncoder.train() 
    Discriminator_1.train()
    Discriminator_2.train()
    Discriminator_3.train()
    ImageEncoder.eval()
        # 创建一个外层的 tqdm 进度条用于 epoch
    epoch_progress = tqdm(range(0, num_epochs), desc="Epochs", unit="epoch")
    torch.autograd.set_detect_anomaly(True)
    G_D_step = 1
    for epoch in epoch_progress:

        num_batches = 0
        g_loss_G_1_mean_max = float('inf')
        g_loss_G_2_mean_max = float('inf')
        d_loss_D_1_mean_max = float('inf')
        d_loss_D_2_mean_max = float('inf')
        g_loss_G_1_sum = 0
        g_loss_G_2_sum = 0
        g_loss_G_3_sum = 0
        d_loss_D_1_sum = 0
        d_loss_D_2_sum = 0
        d_loss_D_3_sum = 0

        batch_progress = tqdm(dataset, desc=f"Epoch {epoch}/{num_epochs}", leave=False, unit="batch")

        for json_path, images, emo, adj, pair in batch_progress:
            
            g_loss_G_1_sum_batch = 0
            g_loss_G_2_sum_batch = 0
            g_loss_G_3_sum_batch = 0
            d_loss_D_1_sum_batch = 0
            d_loss_D_2_sum_batch = 0
            d_loss_D_3_sum_batch = 0
            
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

            g_loss_G_1, g_loss_G_2, g_loss_G_3 = g_loss
            d_loss_D_1, d_loss_D_2, d_loss_D_3 = d_loss  


            # Accumulate generator and discriminator loss
            d_loss_D_3_sum += d_loss_D_3.item()
            d_loss_D_2_sum += d_loss_D_2.item()
            d_loss_D_1_sum += d_loss_D_1.item()
            if num_batches % G_D_step == 0:
                g_loss_G_3_sum += g_loss_G_3.item()
                g_loss_G_2_sum += g_loss_G_2.item()
                g_loss_G_1_sum += g_loss_G_1.item()
            

            # Calculate mean generator loss for the epoch
            d_loss_D_3_sum_batch += d_loss_D_3.item()
            d_loss_D_2_sum_batch += d_loss_D_2.item()
            d_loss_D_1_sum_batch += d_loss_D_1.item()
            if num_batches % G_D_step == 0:
                # g_loss_G_3_sum_batch += g_loss_G_3.item()
                g_loss_G_2_sum_batch += g_loss_G_2.item()
                g_loss_G_1_sum_batch += g_loss_G_1.item()

            d_loss_D_3_mean_batch = d_loss_D_3_sum_batch / batch if batch > 0 else 0.0
            d_loss_D_2_mean_batch = d_loss_D_2_sum_batch / batch if batch > 0 else 0.0
            d_loss_D_1_mean_batch = d_loss_D_1_sum_batch / batch if batch > 0 else 0.0
            if num_batches % G_D_step == 0:
                g_loss_G_3_mean_batch = g_loss_G_3_sum_batch / batch if batch > 0 else 0.0
                g_loss_G_2_mean_batch = g_loss_G_2_sum_batch / batch if batch > 0 else 0.0
                g_loss_G_1_mean_batch = g_loss_G_1_sum_batch / batch if batch > 0 else 0.0
                # print(f"Epoch [{epoch}], g_loss_G_1_mean_batch: {g_loss_G_1_mean_batch:.4f}, g_loss_G_2_mean_batch: {g_loss_G_2_mean_batch:.4f}")

            # print(f"Epoch [{epoch}], d_loss_D_1_mean_batch: {d_loss_D_1_mean_batch:.4f}, d_loss_D_2_mean_batch: {d_loss_D_2_mean_batch:.4f}")
            num_batches += 1
            
        scheduler_1.step()
        scheduler_1.step()
        scheduler_G_1.step()
        scheduler_G_2.step()
        # Calculate mean generator loss for the epoch
        d_loss_D_3_mean = d_loss_D_3_sum / num_batches if num_batches > 0 else 0.0
        d_loss_D_2_mean = d_loss_D_2_sum / num_batches if num_batches > 0 else 0.0
        d_loss_D_1_mean = d_loss_D_1_sum / num_batches if num_batches > 0 else 0.0
        g_loss_G_3_mean = g_loss_G_3_sum / num_batches if num_batches > 0 else 0.0
        g_loss_G_2_mean = g_loss_G_2_sum / num_batches if num_batches > 0 else 0.0
        g_loss_G_1_mean = g_loss_G_1_sum / num_batches if num_batches > 0 else 0.0
        print(f"Epoch [{epoch}/{num_epochs}], d_loss_D_1_mean: {d_loss_D_1_mean:.4f}, d_loss_D_2_mean: {d_loss_D_2_mean:.4f}, d_loss_D_3_mean: {d_loss_D_3_mean:.4f}")
        print(f"Epoch [{epoch}/{num_epochs}], g_loss_G_1_mean: {g_loss_G_1_mean:.4f}, g_loss_G_2_mean: {g_loss_G_2_mean:.4f}, g_loss_G_3_mean: {g_loss_G_3_mean:.4f}")

        # Step the learning rate scheduler
        if (epoch + 1) % 1 == 0 and g_loss_G_1_mean < g_loss_G_1_mean_max:
            g_loss_G_1_mean_max = g_loss_G_1_mean
            g_final_save_path = os.path.join(save_dir,f'graph_encoder_final.pth')
            torch.save(GraphEncoder.state_dict(), g_final_save_path)
            # g_save_path = os.path.join(save_dir,f'graph_encoder_{epoch+1}_{g_loss_G_1_mean}.pth')
        if (epoch + 1) % 1 == 0 and g_loss_G_2_mean < g_loss_G_2_mean_max:
            g_loss_G_2_mean_max = g_loss_G_2_mean
            t_final_save_path = os.path.join(save_dir,f'text_encoder_final.pth')
            torch.save(GraphEncoder.state_dict(), t_final_save_path)
            # t_save_path = os.path.join(save_dir,f'text_encoder_{epoch+1}_{g_loss_G_2_mean}.pth')
        if (epoch + 1) % 1 == 0 and d_loss_D_1_mean < d_loss_D_1_mean_max:
            d_loss_D_1_mean_max = d_loss_D_1_mean
            d1_final_save_path = os.path.join(save_dir,f'discriminator_1_final.pth')
            torch.save(Discriminator_1.state_dict(), d1_final_save_path)
            # d1_save_path = os.path.join(save_dir,f'discriminator_1_{epoch+1}_{d_loss_D_1_mean}.pth')
        if (epoch + 1) % 1 == 0 and d_loss_D_2_mean < d_loss_D_2_mean_max:
            d_loss_D_2_mean_max = d_loss_D_2_mean
            d2_final_save_path = os.path.join(save_dir,f'discriminator_2_final.pth')
            torch.save(Discriminator_2.state_dict(), d2_final_save_path)
            # d2_save_path = os.path.join(save_dir,f'discriminator_2_{epoch+1}_{d_loss_D_2_mean}.pth')
    


# 训练函数
def train():
    gpu_id = 1
    batch_size = 4
    num_epochs = 4
    learning_rate = 5e-6
    proportion = 1
    save_dir = "/home/zjx/lab/my3_lab/v1/checkpoint"
    dataloader = get_dataset(batch_size,proportion)  # 获取数据集
    ImageEncoder, GraphEncoder, TextEncoder, Discriminator_graph_img, Discriminator_text_img, Discriminator_graph_text, alpha_logit = creat_model(gpu_id)
    model = ImageEncoder, GraphEncoder, TextEncoder, Discriminator_graph_img, Discriminator_text_img, Discriminator_graph_text, alpha_logit
    device = torch.device(f"cuda:{gpu_id}")
    train_emotion_token_module(model, dataloader,device,save_dir,learning_rate=learning_rate,batch_size=batch_size, num_epochs=num_epochs)

if __name__ == "__main__":
    train()
