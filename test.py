import torch
import torch.nn as nn
import math

def initialize_target_mean(key, dim=8, scale=10.0):
    N = 8
    idx = emotion_keys.index(key)
    theta = 2 * math.pi * idx / N
    vector = torch.zeros(dim)
    vector[0] = math.cos(theta) * scale
    vector[1] = math.sin(theta) * scale
    if dim > 2:
        vector[2:] = torch.randn(dim-2) * 0.1  # 小幅度随机噪声
    return vector
# 示例用法
emotion_keys = ['amusement', 'contentment', 'awe', 'excitement', 'fear', 'sadness', 'disgust', 'anger']
for key in emotion_keys:
    means = initialize_target_mean(key, dim=8)
    print(means)
