from model.vgg import vgg as vgg_model
from model.decoder import decoder as decoder_model
import model.clip_emo_classifier as clip_emo_classifier
import model.transformer as transformer
import model.EmoTransModel_test as big_Model
from model.EmoTransModel import MLP as MLP
from model.EmoTransModel import Transform as Transform
from model.discriminator import Discriminator as Discriminator
import clip
import emo_token_gen_model.emo_token_gen_model as emotion_space
from data.emosetplus_content_test import EmoAdjDataset_content
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
from torchvision.utils import save_image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

iter_num = 40000 # 5000

def set_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--emo_conv', type=str, default=f"/home/zjx/lab/my3_lab/v1/saved_nerwork/emo_conv_{iter_num}.pth")  #run the train.py, please download the pretrained VAD_emb checkpoint
    parser.add_argument('--Mapper', type=str, default=f"/home/zjx/lab/my3_lab/v1/saved_nerwork/Mapper_iter_{iter_num}.pth")  #run the train.py, please download the pretrained VAD_emb checkpoint
    parser.add_argument('--vgg', type=str, default='/home/zjx/lab/my3_lab/v1/pretrained_model/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint
    parser.add_argument('--PatchEmbed', type=str, default=f"/home/zjx/lab/my3_lab/v1/saved_nerwork/patchembed_iter_{iter_num}.pth")  #run the train.py, please download the pretrained embedding checkpoint
    parser.add_argument('--Trans', type=str, default=f"/home/zjx/lab/my3_lab/v1/saved_nerwork/transformer_iter_{iter_num}.pth")  #run the train.py, please download the pretrained Trans checkpoint
    parser.add_argument('--decoder', type=str, default=f"/home/zjx/lab/my3_lab/v1/saved_nerwork/decoder_iter_{iter_num}.pth")  #run the train.py, please download the pretrained decoder checkpoint
    parser.add_argument('--embedding_group', type=str, default="/home/zjx/lab/my3_lab/v1/checkpoint/embedding_group_final.pth")  #run the train.py, please download the pretrained decoder checkpoint
    parser.add_argument('--save_dir', default='/home/zjx/lab/my3_lab/v1/saved_imgs',help='Directory to save the model')
    parser.add_argument('--save_net_dir', default='/home/zjx/lab/my3_lab/v1/saved_nerwork',help='Directory to save the model')
    parser.add_argument('--content_img_path', default='/home/zjx/lab/dataset/Artphoto/train/fear/fear_0682.jpg',help='Directory to save the model')
    parser.add_argument('--content_img_save_path', default='/home/zjx/lab/my3_lab/v1/single_img/single_img.jpg',help='Directory to save the model')
    parser.add_argument('--content_img_save_dir', default='/home/zjx/lab/my3_lab/v1/single_img',help='Directory to save the model')
    parser.add_argument('--content_img_output', default='/home/zjx/lab/my3_lab/v1/output2',help='Directory to save the model')
    # parser.add_argument('--save_net_dir', default='/home/zjx/lab/my3_lab/v1/saved_nerwork',help='Directory to save the model')
    
    parser.add_argument('--mode', type=str, default="b",help="s/b")
    parser.add_argument('--text', type=str, default="amusement")
    parser.add_argument('--index', type=int, default=1,help="0-255") 
    parser.add_argument('--loaded', type=bool, default=True)
    parser.add_argument('--gpuid', type=int, default=3)
    parser.add_argument('--c_proportion', type=float, default=1)
    parser.add_argument('--proportion', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3047)
    args = parser.parse_args()
    return args

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

    # load text_encoder
    clip_net, _ = clip.load("ViT-B/16",device=device,jit=False)

    emo_conv = nn.Conv2d(kernel_size=(1,1),stride=1,in_channels=512,out_channels=512).to(device)
    emo_conv = load_model(emo_conv, args.emo_conv, device)

    Mapper = MLP(device=device)
    Mapper = load_model(Mapper, args.Mapper, device)

    pretrain_model = (vgg, decoder, PatchEmbed, Trans, emb_group, clip_net, emo_conv, Mapper)

    return pretrain_model

# resize content picture
def content_transform():
    c_mean = (0.5,0.5,0.5)
    c_std = (0.5,0.5,0.5)
    # c_mean = (0.52, 0.465, 0.40)
    # c_std = (0.22, 0.21,0.19)
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

def get_dataset(dataset_type, trans_type, batch_size=1, c_proportion=0.5, proportion=0.9):
    image_root = '/home/public/datasets/EmoSet-118K/image'
    json_root = '/home/zjx/lab/dataset/emoset_v1'
    s_proportion = 1 - c_proportion
    if dataset_type == 'content':
        dataset = EmoAdjDataset_content(img_root=image_root,
                                json_root=json_root,
                                c_proportion=c_proportion,
                                proportion=proportion,
                                is_train=False,
                                transform=trans_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
        # sampler=InfiniteSamplerWrapper(dataset)
    else:
        dataset = EmoAdjDataset_style(img_root=image_root,
                        json_root=json_root,
                        s_proportion=s_proportion,
                        proportion=proportion,
                        is_train=False,
                        transform=trans_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    
    return dataloader

def find_target_text(ori_emotion):
    emotion_ori_target_dict = {
        'amusement':'fear',
        'contentment':'sadness',
        'awe':'disgust',
        'excitement':'anger',
        'fear':'amusement',
        'sadness':'contentment',
        'disgust':'awe',
        'anger':'excitement'
    }
    target_emotion = emotion_ori_target_dict[ori_emotion]
    return target_emotion

def test():
    args = set_args()
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.gpuid}")
    dtype = torch.float32
    print("device:",args.gpuid)

    pretrain_model = pretrain_model_load(args, device)
    vgg, decoder, PatchEmbed, Trans, embedding_group, clip_net,  emo_conv, Mapper = pretrain_model

    vgg.eval()
    embedding_group.eval()
    clip_net.eval()
    PatchEmbed.eval()
    Trans.eval()
    emo_conv.eval()
    decoder.eval()
    Mapper.eval()

    network = big_Model.EmoTransModel_test(vgg,
                                decoder,
                                PatchEmbed,
                                Trans,
                                embedding_group,
                                clip_net,
                                emo_conv,
                                Mapper,
                                device)
    network.eval()
    network.to(device)

    c_tf = content_transform()

    # s_mean = (0.5,0.5,0.5)
    # s_std = (0.5,0.5,0.5)
    s_mean = (0.52, 0.465, 0.40)
    s_std = (0.22, 0.21,0.19)
    std_s = torch.tensor(s_std).view(1, -1, 1, 1).to(device)
    mean_s = torch.tensor(s_mean).view(1, -1, 1, 1).to(device)

    if args.mode == 's':
        content_path = args.content_img_path
        content = c_tf(Image.open(content_path).convert("RGB"))
        content = content.to(device).unsqueeze(0)
        content = content * std_s + mean_s
        with torch.no_grad():
            output= network(content, args.text, args.index)
        output = output * std_s + mean_s
        output = output.to(device)
        output = torch.cat([content,output],dim=0)
        output_path = args.content_img_save_path
        save_image(output, output_path)      
    elif args.mode == 'b':
        # read content_dataset
        content_dataloader = get_dataset("content",c_tf, 1, args.c_proportion, args.proportion)
        content_iter = iter(content_dataloader)
        dataloader_len = len(content_dataloader)
        for i in tqdm(range(dataloader_len)):
            
            content_image, img_path = next(content_iter)
            img_path = img_path[0]
            content_image = content_image.to(device).to(dtype)
            file_name = os.path.basename(img_path)
            ori_emotion = file_name.split('_')[0].lower()
            target = find_target_text(ori_emotion)
            # import pdb;pdb.set_trace()
            with torch.no_grad():
                output = network(content_image, target)
            output = output * std_s + mean_s
            output = output.to(device)
            output_dir = args.content_img_output
            output_path = os.path.join(output_dir, f'target_{target}_{file_name}')
            # output = output * std_s + mean_s
            # output = output.cpu()
            save_image(output, output_path)
        

if __name__ == "__main__": 
    test()
