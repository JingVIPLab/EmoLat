import os
from PIL import Image
import numpy as np
import clip
from loguru import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import os
import json
class EmoAdjDataset(Dataset):
    def __init__(self,img_root,json_root, proportion,is_train,preprocess):
        # 0.比例
        self.proportion = proportion
        # 1.根目录(根据自己的情况更改)
        self.img_root = img_root
        self.json_root = json_root
        # 2.训练图片和测试图片地址(根据自己的情况更改)

        self.train_set_file = []
        self.test_set_file = []
        emo_class = ['amusement','anger','awe','contentment','disgust','excitement','fear','sadness']
        # 每个类别都筛选出一定比例
        for emo_cls in emo_class:
            emo_json_root = os.path.join(json_root,emo_cls)
            json_cls_file = []
            for root, dirs, files in os.walk(emo_json_root):
                for file in files:
                    json_path = os.path.join(root,file)
                    json_cls_file.append(json_path)
            split_index = int(len(json_cls_file) * self.proportion)
            train_cls_set_file = json_cls_file[:split_index]
            test_cls_set_file = json_cls_file[split_index:]
            self.train_set_file.extend(train_cls_set_file)
            self.test_set_file.extend(test_cls_set_file)

        
        # 3.训练 or 测试(根据自己的情况更改)
        self.is_train = is_train
        # 4.处理图像
        self.img_process = preprocess
        # 5.获得数据(根据自己的情况更改)
        self.samples = []
        self.sam_labels = []
        # 5.1 训练还是测试数据集
        self.read_file = []
        if is_train:
            self.read_file = self.train_set_file
        else:
            self.read_file = self.test_set_file
		# 5.2 获得所有的样本(根据自己的情况更改)
        # read_file是json路径list
        for json_path in self.read_file:
            with open(json_path) as json_file:
                json_data = json.load(json_file)
            img_path = json_path.replace(self.json_root,self.img_root).replace('.json','.jpg')
            label = json_data["emotion"]
            label = f'a photo seems to be {label}'
            self.samples.append(img_path)
            self.sam_labels.append(label)
        self.tokens = clip.tokenize(self.sam_labels)

        # with open(self.read_file,'r') as f:
        #     for line in f:
        #         img_path = os.path.join(self.img_root,line.strip() + '.jpg')
        #         label = line.strip().split('/')[0]
        #         label = label.replace("_"," ")
        #         label = "photo if " + label
        #         self.samples.append(img_path)
        #         self.sam_labels.append(label)
        # # 转换为token
        # self.tokens = clip.tokenize(self.sam_labels)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        token = self.tokens[idx]
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        # 对图像进行转换
        image = self.img_process(image)
        return image, token
# set_model
# 创建模型
class ClipModel(nn.Module):
    def __init__(self, gpu_id):
        super(ClipModel, self).__init__() 
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    def creat_model(self):
        self.net, self.preprocess = clip.load("ViT-B/16",device=self.device,jit=False)
        self.optimizer = optim.Adam(self.net.parameters(), lr=5e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        print("successful load clip model")
        return self.net, self.preprocess, self.optimizer,self.scheduler

    # 创建损失函数
    def loss(self):
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        return loss_img, loss_txt

def train():
    # 加载数据集
    image_root = '/home/public/datasets/EmoSet-118K/image'
    json_root = '/home/zjx/lab/dataset/emoset_v1'
    device = 0
    clip_model = ClipModel(device)
    net, preprocess, optimizer, scheduler = clip_model.creat_model()
    loss_img, loss_txt = clip_model.loss()
    dataset = EmoAdjDataset(img_root=image_root,
                                json_root=json_root,
                                proportion=1,
                                is_train=True,
                                preprocess=preprocess)
    checkpoint_save_dir = '/home/zjx/lab/test/clip_tuning/checkpoint'

    
    
    dataset_size = len(dataset)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4,pin_memory=False)

    phase = "train"
    model_name = "clip_tuning_model_v3_ViT-B16"
    ckt_gap = 5
    epoches = 50
    for epoch in range(epoches):
        scheduler.step()
        total_loss = 0
        batch_num = 0
        # 使用混合精度，占用显存更小
        with torch.cuda.amp.autocast(enabled=True):
            for images,label_tokens in dataloader:
                # 将图片和标签token转移到device设备
                images = images.to(f'cuda:{device}')
                # import pdb;pdb.set_trace()
                label_tokens = label_tokens.to(f'cuda:{device}')
                batch_num += 1
                # 优化器梯度清零
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits_per_image, logits_per_text = net(images, label_tokens)
                    ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                    cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                    total_loss += cur_loss
                    if phase == "train":
                        cur_loss.backward()
                        if device == "cpu":
                            optimizer.step()
                        else:
                            optimizer.step()
                            clip.model.convert_weights(net) 
                if batch_num % 4 == 0:
                    logger.info('{} epoch:{} loss:{}'.format(phase,epoch,cur_loss))
                    epoch_loss = total_loss / dataset_size
                    epoch_save_dir = '/home/zjx/lab/test/clip_tuning'
                    epoch_save_path = os.path.join(epoch_save_dir, model_name)
                    os.makedirs(epoch_save_path, exist_ok=True)
                    torch.save(net.state_dict(),os.path.join(epoch_save_path,f"{model_name}_epoch_{epoch}.pth"))
                    logger.info(f"weights_{epoch} saved")
            if epoch % ckt_gap == 0:
                # checkpoint_path = f"{model_name}_ckt.pth"
                checkpoint_path = os.path.join(checkpoint_save_dir,f"{model_name}_ckt.pth")
                checkpoint = {
                    'it': epoch,
                    'network': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")
            logger.info('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

if __name__=="__main__":
    train()