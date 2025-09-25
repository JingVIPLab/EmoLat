import os
from PIL import Image
import numpy as np
from loguru import logger
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import os
import json
from torchvision import transforms

class EmoAdjDataset(Dataset):
    def __init__(self, img_root, json_root, proportion, is_train):
        # 0.比例
        self.proportion = proportion
        # 1.根目录(根据自己的情况更改)
        self.img_root = img_root
        self.json_root = json_root
        self.train_set_file = []
        self.test_set_file = []
        emo_class = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
        # 每个类别都筛选出80%
        for emo_cls in emo_class:
            emo_json_root = os.path.join(json_root, emo_cls)
            json_cls_file = []
            for root, dirs, files in os.walk(emo_json_root):
                for file in files:
                    json_path = os.path.join(root, file)
                    json_cls_file.append(json_path)
            # import pdb;pdb.set_trace()
            split_start_index = int(len(json_cls_file) * (self.proportion-0.2))        
            split_end_index = int(len(json_cls_file) * self.proportion)
            
            # import pdb;pdb.set_trace()
            # split_index = int(len(json_cls_file) * self.proportion)
            # train_cls_set_file = json_cls_file[:split_index]

            train_cls_set_file = json_cls_file[split_start_index: split_end_index]
            test_cls_set_file = json_cls_file[split_start_index:]
            self.train_set_file.extend(train_cls_set_file)
            self.test_set_file.extend(test_cls_set_file)

        # 3.训练 or 测试(根据自己的情况更改)
        self.is_train = is_train

        # 4.图像变换定义
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # 5.获得数据(根据自己的情况更改)
        self.samples = []
        self.samples_labels_pair = []
        self.samples_labels_adj = []
        self.samples_labels_emo = []

        # 5.1 训练还是测试数据集
        self.read_file = []
        if is_train:
            self.read_file = self.train_set_file
        else:
            self.read_file = self.test_set_file

        # 5.2 获得所有的样本(根据自己的情况更改)
        # read_file是json路径list
        max_len = 14
        for json_path in self.read_file:
            with open(json_path) as json_file:
                json_data = json.load(json_file)
            img_path = json_path.replace(self.json_root, self.img_root).replace('.json', '.jpg')
            # import pdb;pdb.set_trace()
            global_adj_label = json_data["global_adjective"]
            obj_adj_label = json_data["ontology_adjective"]
            add_len = max_len - len(obj_adj_label)
            if add_len == 14:
                continue
            if add_len <= 0:
                break
            for _ in range(add_len):
                obj_adj_label.append(["<pad>", "<pad>"])
            emo_label = json_data["emotion"]

            self.samples.append(img_path)
            self.samples_labels_pair.append(obj_adj_label)
            self.samples_labels_emo.append(emo_label)
            self.samples_labels_adj.append(global_adj_label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        adj = self.samples_labels_adj[idx]
        emo = self.samples_labels_emo[idx]
        pair = self.samples_labels_pair[idx]

        # 加载图像并应用变换
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # 对图像进行转换

        json_path = img_path.replace("/home/public/datasets/EmoSet-118K/image","/home/zjx/lab/dataset/emoset_v1").replace('.jpg','.json')
        return json_path, image, emo, adj, pair