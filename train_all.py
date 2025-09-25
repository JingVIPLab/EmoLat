from model.vgg import vgg as vgg_model
from model.decoder import decoder as decoder_model
import model.clip_emo_classifier as clip_emo_classifier
import model.transformer as transformer
import model.EmoTransModel as big_Model
from model.EmoTransModel import MLP as MLP
from model.EmoTransModel import Transform as Transform
from model.discriminator import Discriminator as Discriminator
import clip
import emo_token_gen_model.emo_token_gen_model as emotion_space
from data.emosetplus_content import EmoAdjDataset_content
from data.emosetplus_style import EmoAdjDataset_style
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import random
import numpy as np
from torchvision import transforms
from model.sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageDraw, ImageFont
import re
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None



# norm
# c_mean = (0.485,0.456,0.406)
# c_std = (0.229,0.224,0.225)
# s_mean = (0.485,0.456,0.406)
# s_std = (0.229,0.224,0.225)

s_mean = (0.52, 0.465, 0.40)
s_std = (0.22, 0.21,0.19)
c_mean =  (0.52, 0.465, 0.40)
c_std = (0.22, 0.21,0.19)


def set_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--SV_1', type=str, default='/home/zjx/lab/my3_lab/v1/pretrained_model/SV_1.pth')
    parser.add_argument('--SV_2', type=str, default='/home/zjx/lab/my3_lab/v1/pretrained_model/SV_2.pth')
    parser.add_argument('--SV_3', type=str, default='/home/zjx/lab/my3_lab/v1/pretrained_model/SV_3.pth')
    parser.add_argument('--SV_4', type=str, default='/home/zjx/lab/my3_lab/v1/pretrained_model/SV_4.pth')
    parser.add_argument('--SV_5', type=str, default='/home/zjx/lab/my3_lab/v1/pretrained_model/SV_5.pth')
    parser.add_argument('--VAD_emb', type=str, default="/home/zjx/lab/my3_lab/v1/pretrained_model/VAD_emb.pth")  #run the train.py, please download the pretrained VAD_emb checkpoint
    parser.add_argument('--vgg', type=str, default='/home/zjx/lab/my3_lab/v1/pretrained_model/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint
    parser.add_argument('--PatchEmbed', type=str, default="/home/zjx/lab/my3_lab/v1/pretrained_model/embedding.pth")  #run the train.py, please download the pretrained embedding checkpoint
    parser.add_argument('--Trans', type=str, default="/home/zjx/lab/my3_lab/v1/pretrained_model/transformer.pth")  #run the train.py, please download the pretrained Trans checkpoint
    parser.add_argument('--decoder', type=str, default="/home/zjx/lab/my3_lab/v1/pretrained_model/decoder.pth")  #run the train.py, please download the pretrained decoder checkpoint
    # parser.add_argument('--emo_encoder', type=str, default="/home/zjx/lab/my3_lab/v1/checkpoint/text_encoder_final.pth")  #run the train.py, please download the pretrained decoder checkpoint
    parser.add_argument('--gragh_encoder', type=str, default="/home/zjx/lab/my3_lab/v1/checkpoint/graph_encoder_final.pth")  #run the train.py, please download the pretrained decoder checkpoint
    parser.add_argument('--embedding_group', type=str, default="/home/zjx/lab/my3_lab/v1/checkpoint/embedding_group_final.pth")  #run the train.py, please download the pretrained decoder checkpoint
    parser.add_argument('--means', type=str, default="/home/zjx/lab/my3_lab/v1/checkpoint/mean_final.pth")  #run the train.py, please download the pretrained decoder checkpoint
    parser.add_argument('--std', type=str, default="/home/zjx/lab/my3_lab/v1/checkpoint/std_final.pth")  #run the train.py, please download the pretrained decoder checkpoint
    parser.add_argument('--graph_disciminator', type=str, default="/home/zjx/lab/my3_lab/v1/checkpoint/discriminator_final.pth")  #run the train.py, please download the pretrained decoder checkpoint
    parser.add_argument('--clip_emo_classifier', type=str, default="/home/zjx/lab/clip_tuning/clip_tuning_model_v3_ViT-B16/clip_tuning_model_v3_ViT-B16_epoch_25.pth")  #run the train.py, please download the pretrained decoder checkpoint
    parser.add_argument('--save_dir', default='/home/zjx/lab/my3_lab/v1/saved_imgs',help='Directory to save the model')
    parser.add_argument('--save_net_dir', default='/home/zjx/lab/my3_lab/v1/saved_nerwork',help='Directory to save the model')
    
    parser.add_argument('--loaded', type=bool, default=True)
    parser.add_argument('--gpuid', type=int, default=2)
    parser.add_argument('--c_proportion', type=float, default=0.5)
    parser.add_argument('--proportion', type=float, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--show_pic', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--style_weight', type=float, default=2)
    parser.add_argument('--content_weight', type=float, default=6)
    parser.add_argument('--l_identity1', type=float, default=3.0)
    parser.add_argument('--l_identity2', type=float, default=1)
    parser.add_argument('--tv_loss', type=float, default=1.0)
    parser.add_argument('--homo_loss', type=float, default=30.0)
    parser.add_argument('--clip_loss', type=float, default=200.0)
    parser.add_argument('--graph_loss', type=float, default=1.0)
    parser.add_argument('--D_loss', type=float, default=5.0)
    parser.add_argument('--G_loss', type=float, default=5.0)
    parser.add_argument('--directional_loss', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    args = parser.parse_args()
    return args

# define seed
def setup_seed(seed):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

def pretrain_model_load(args,device):
    # load vgg
    vgg = vgg_model
    vgg = load_model(vgg, args.vgg, device)
    vgg = nn.Sequential(*list(vgg.children())[:44])

    # load decoder
    decoder = decoder_model
    decoder = load_model(decoder, args.decoder, device)

    # load PatchEmbedd
    PatchEmbed = big_Model.PatchEmbed()
    PatchEmbed = load_model(PatchEmbed, args.PatchEmbed, device)

    # load transformer
    Trans = transformer.Transformer()
    Trans = load_model(Trans, args.Trans, device)

    # load embedding_group means and std 
    emb_model = emotion_space.EmbeddingGroup(device=device)
    emb_group = load_model(emb_model.embedding_group, args.embedding_group, device)
    emb_means = load_model(emb_model.means, args.means, device)
    emb_std = load_model(emb_model.stds, args.std, device)

    # load graph_encoder
    # graph_encoder = emotion_space.GraphEncoder(512,512,512,512,512,device)
    # graph_encoder = load_model(graph_encoder, args.gragh_encoder, device)

    # load text_encoder
    clip_net, _ = clip.load("ViT-B/16",device=device,jit=False)

    # load discriminator
    # graph_disciminator = emotion_space.Discriminator(device)
    # graph_disciminator = load_model(graph_disciminator, args.graph_disciminator, device)

    # load clip_classify
    clip_classifier = clip_emo_classifier.ClipModel(args.gpuid)
    clip_classifier, _, _, _ = clip_classifier.creat_model()
    clip_classifier.load_state_dict(torch.load(args.clip_emo_classifier))
    clip_classifier.to(device)

    # load_sentiment_vector
    emo_1 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=64,out_channels=16)
    emo_2 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=128,out_channels=16)
    emo_3 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=256,out_channels=16)
    emo_4 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=512,out_channels=16)
    emo_5 = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=512,out_channels=16)
    emo_1.load_state_dict(torch.load(args.SV_1))
    emo_2.load_state_dict(torch.load(args.SV_2))
    emo_3.load_state_dict(torch.load(args.SV_3))
    emo_4.load_state_dict(torch.load(args.SV_4))
    emo_5.load_state_dict(torch.load(args.SV_5))
    emo_1.to(device)
    emo_2.to(device)
    emo_3.to(device)
    emo_4.to(device)
    emo_5.to(device)

    sentiment_vector = (emo_1,emo_2,emo_3,emo_4,emo_5)

    emo_conv = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=512,out_channels=512).to(device)
    # emo_conv.load_state_dict(torch.load(args.VAD_emb))

    Mapper = MLP(device=device)

    # style_net = StyleNet.UNet()
    graph_encoder = None
    graph_disciminator = None

    pretrain_model = (vgg, decoder, PatchEmbed, Trans, emb_group, emb_means, emb_std, clip_net, graph_encoder, graph_disciminator, clip_classifier, sentiment_vector, emo_conv, Mapper)

    return pretrain_model

# resize content picture
def content_transform():
    # c_mean = (0.5,0.5,0.5)
    # c_std = (0.5,0.5,0.5)

    c_mean = (0.52, 0.465, 0.40)
    c_std = (0.22, 0.21,0.19)

    # c_mean = (0.485,0.456,0.406)
    # c_std = (0.229,0.224,0.225)
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=c_mean, std=c_std)
    ]
    return transforms.Compose(transform_list)

# resize style picture
def style_transform():
    s_mean = (0.52, 0.465, 0.40)
    s_std = (0.22, 0.21,0.19)
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=s_mean, std=s_std)
    ]
    return transforms.Compose(transform_list)

# adjust learning rate1
def adjust_learning_rate(args, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# adjust learning rate2
def warmup_learning_rate(args, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_dataset(dataset_type, trans_type, batch_size=1, c_proportion=0.5, proportion=0.9):
    image_root = '/home/public/datasets/EmoSet-118K/image'
    json_root = '/home/zjx/lab/dataset/emoset_v1'
    s_proportion = 1.0 - c_proportion
    if dataset_type == 'content':
        dataset = EmoAdjDataset_content(img_root=image_root,
                                json_root=json_root,
                                c_proportion=c_proportion,
                                proportion=proportion,
                                is_train=True,
                                transform=trans_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=InfiniteSamplerWrapper(dataset), num_workers=4, pin_memory=False)
        # sampler=InfiniteSamplerWrapper(dataset)
    else:
        dataset = EmoAdjDataset_style(img_root=image_root,
                        json_root=json_root,
                        s_proportion=s_proportion,
                        proportion=proportion,
                        is_train=True,
                        transform=trans_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=InfiniteSamplerWrapper(dataset), num_workers=4, pin_memory=False)
    
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

def create_graph_feat(json_path, images, emo, adj, pair, batch_size):
    emo = [[e] for e in emo]
    adj = [[a] for a in adj]
    pair_list = []
    for b in range(batch_size):
        formatted_data = [[item[b] for item in p] for p in pair]
        pair_list.append(formatted_data)
    pair = pair_list
    pair = pair_filter(pair_list)
    object_words_batch, adjective_words_batch, sentiment_word_batch, global_adj_word_batch = change_label_form(emo,adj,pair)
    data = (object_words_batch, adjective_words_batch, sentiment_word_batch, global_adj_word_batch)
    return data


def main():
    # set paraser
    args = set_args()
    # setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpuid}")
    print("device:",args.gpuid)
    # set model
    pretrain_model = pretrain_model_load(args, device)
    # with torch.no_grad():
    vgg, decoder, PatchEmbed, Trans, embedding_group, emb_means, emb_std, clip_net, graph_encoder, graph_disciminator, clip_emo_classifier, sentiment_vector, emo_conv, Mapper = pretrain_model
    vgg.eval()
    embedding_group.eval()
    emb_means.eval()
    emb_std.eval()
    clip_net.eval()
    clip_emo_classifier.eval()
    for item in sentiment_vector:
        item.eval()

    PatchEmbed.train()
    Trans.train()
    emo_conv.train()
    decoder.train()
    Mapper.train()
    
    
    # Trans = big_Model.Transform
    network = big_Model.EmoTransModel(vgg,
                                    decoder,
                                    PatchEmbed,
                                    Trans,
                                    embedding_group,
                                    emb_means, 
                                    emb_std,
                                    clip_net,
                                    graph_encoder,
                                    clip_emo_classifier,
                                    graph_disciminator,
                                    sentiment_vector,
                                    emo_conv,
                                    Mapper,
                                    device)
    # network.train()
    network.to(device)
    
    # get dataset
    c_tf = content_transform()
    s_tf = style_transform()

    # read content_dataset
    content_dataloader = get_dataset("content",c_tf, args.batch_size, args.c_proportion, args.proportion)
    content_iter = iter(content_dataloader)

    # read style dataset
    style_dataloader = get_dataset("style", s_tf, args.batch_size, args.c_proportion, args.proportion)
    style_iter = iter(style_dataloader)

    # define Adam optimizer
    optimizer = torch.optim.Adam([
                                {'params': network.transformer.parameters()},
                                {'params': network.decoder.parameters()},
                                {'params': network.embedding.parameters()},
                                {'params': network.emo_conv.parameters()},
                                {'params': network.Mapper.parameters()}
                                ], lr=args.lr)

    # 定义鉴别器和鉴别器优化器
    discriminator = Discriminator(device=device)
    optim_D = torch.optim.AdamW(discriminator.parameters(), lr=0.00001, betas=(0.9, 0.98))

    # training begin
    for i in tqdm(range(args.max_iter)):
        # define learning rate
        if i < 1e4:
            warmup_learning_rate(args,optimizer, iteration_count=i)
        else:
            adjust_learning_rate(args, optimizer, iteration_count=i)
        print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))

        # json_path, image, emo, adj, pair
        # import pdb;pdb.set_trace()
        content_images, content_path = next(content_iter)
        content_images = content_images.to(device).to(torch.float32)
        style_all = next(style_iter) 

        style_path = style_all[0]
        style_images = style_all[1].to(device)
        style_emo = style_all[2]
        style_adj = style_all[3]
        style_pair = style_all[4]
        style_path = style_all[5]

        text_tensor = torch.tensor([]).to(device)

        graph_data = create_graph_feat(style_path, style_images, style_emo, style_adj, style_pair, args.batch_size)

        _, _, sentiment_word_batch, _ = graph_data
        # import pdb;pdb.set_trace()
        # with torch.no_grad():
        #     emo_text_embedding = emo_encoder(sentiment_word_batch)

        out, loss_c, loss_s, l_identity1, l_identity2, loss_tv, loss_homo, loss_clip, loss_graph, loss_directional, GD = network(content_images, style_images, sentiment_word_batch, graph_data)
        style_feats, fusion_img_feats, text_feats_clip = GD


        # 鉴别器损失
        #########################
        # import pdb;pdb.set_trace()

        style_feats = style_feats
        Ics_feats = fusion_img_feats
        for style_i in range(len(style_feats)):
            style_feats[style_i] = style_feats[style_i].detach()
        for Ics_i in range(len(Ics_feats)):
            Ics_feats[Ics_i] = Ics_feats[Ics_i].detach()
            
        ins = text_feats_clip.detach()
        loss_D = discriminator(style_feats, ins, True) + discriminator(Ics_feats, ins, False)
        loss_D = loss_D * args.D_loss
        # print('D',loss_D.sum().cpu().detach().numpy())
        loss_D.sum().backward()
        optim_D.step()
        Ics_feats_G = fusion_img_feats
        ins_G = text_feats_clip
        loss_G = discriminator(Ics_feats_G,ins_G,True)
        loss_psd = args.G_loss * loss_G

        ###########################

        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        
        l_identity1 = l_identity1 * args.l_identity1
        l_identity2 = l_identity2 * args.l_identity2

        loss_tv = args.tv_loss * loss_tv
        loss_homo = args.homo_loss * loss_homo
        # loss_clip = args.clip_loss * loss_clip
        # loss_graph = 0 # args.graph_loss * loss_graph
        loss_directional = args.directional_loss * loss_directional

        # loss = loss_c + loss_s + l_identity1 + l_identity2 + loss_tv + loss_homo + loss_clip + loss_directional # + loss_graph
        loss = loss_c + loss_s + l_identity1 + l_identity2 + loss_directional + loss_psd + loss_tv + loss_homo
        print("-content:",loss_c.sum().cpu().detach().numpy(),
              "-style:",loss_s.sum().cpu().detach().numpy(),
              "-l1:",l_identity1.sum().cpu().detach().numpy(),
              "-l2:",l_identity2.sum().cpu().detach().numpy(),
              # "-TV:",loss_tv.sum().cpu().detach().numpy(),
              # "-loss_homo:",loss_homo.sum().cpu().detach().numpy(),
              # "-loss_clip:",loss_clip.sum().cpu().detach().numpy(),
              # "-loss_graph:",loss_graph.sum().cpu().detach().numpy()
              "-loss_directional:",loss_directional.sum().cpu().detach().numpy(),
              "-loss_psd:",loss_psd.sum().cpu().detach().numpy()
            )

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if i % args.show_pic == 0:
            output_name = os.path.join(args.save_dir,f"{str(i)}.jpg")
            white_images = torch.ones(min(4,args.batch_size),3,256,256).to(device)
            white_images = white_images * 255

            std_c = torch.tensor(c_std).view(1, -1, 1, 1).to(device)
            mean_c = torch.tensor(c_mean).view(1, -1, 1, 1).to(device)
            content_images = content_images * std_c + mean_c
            content_images = content_images[:min(4,args.batch_size), :, :, :]
            out = out[:min(4,args.batch_size), :, :, :]
            out = torch.cat((content_images,out),0)
            out = torch.cat((white_images,out),0)
            save_image(out, output_name)

            img = Image.open(output_name)
            draw = ImageDraw.Draw(img)
            typeface = ImageFont.truetype("/home/zjx/lab/my3_lab/v1/Arial.ttf", 30)
            for i in range(min(4,args.batch_size)):
                miaoshu = style_emo[i]
                miaoshu = re.sub(r"(.{30})", "\\1\n", miaoshu)
                draw.text((i * 258, 10), miaoshu, fill=(120, 0, 60), font=typeface)
            img.save(output_name)

        # save model
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:

            state_dict = network.emo_conv.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                        '{:s}/emo_conv_{:d}.pth'.format(args.save_net_dir,
                                                                i + 1))

            state_dict = network.transformer.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                    '{:s}/transformer_iter_{:d}.pth'.format(args.save_net_dir,
                                                            i + 1))

            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                    '{:s}/decoder_iter_{:d}.pth'.format(args.save_net_dir,
                                                            i + 1))
            state_dict = network.embedding.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                    '{:s}/patchembed_iter_{:d}.pth'.format(args.save_net_dir,
                                                            i + 1))
            
            state_dict = network.Mapper.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                    '{:s}/Mapper_iter_{:d}.pth'.format(args.save_net_dir,
                                                            i + 1))

if __name__ == "__main__":
    main()