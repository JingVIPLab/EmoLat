import torch
import torch.nn as nn
class Discriminator(torch.nn.Module): 
    def __init__(self, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # self.text_linear = MLP(40,1)

        self.bce_loss = nn.BCELoss()

        self.fc_0 = torch.nn.Sequential(*[ torch.nn.Linear(64+64+512, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype)

        self.fc_1 = torch.nn.Sequential(*[ torch.nn.Linear(128+128+512, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype)

        self.fc_2 = torch.nn.Sequential(*[ torch.nn.Linear(256+256+512, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype)
                     
        self.fc_3 = torch.nn.Sequential(*[ torch.nn.Linear(512+512+512, 512), nn.BatchNorm1d(512), torch.nn.ReLU(),
                                    torch.nn.Linear(512, 128), nn.BatchNorm1d(128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype) 

        self.fc_4 = torch.nn.Sequential(*[ torch.nn.Linear(512+512+512, 512), nn.BatchNorm1d(512), torch.nn.ReLU(),
                                    torch.nn.Linear(512, 128), nn.BatchNorm1d(128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype)

        self.fc_un_0 = torch.nn.Sequential(*[ torch.nn.Linear(128, 32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype)
        self.fc_un_1 = torch.nn.Sequential(*[ torch.nn.Linear(256, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype)
        self.fc_un_2 = torch.nn.Sequential(*[ torch.nn.Linear(512, 128), nn.BatchNorm1d(128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 32), nn.BatchNorm1d(32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype)
        self.fc_un_3 = torch.nn.Sequential(*[ torch.nn.Linear(1024, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype)
        self.fc_un_4 = torch.nn.Sequential(*[ torch.nn.Linear(1024, 256), nn.BatchNorm1d(256), torch.nn.ReLU(),
                                    torch.nn.Linear(256, 64), nn.BatchNorm1d(64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Sigmoid()]).to(self.device).to(self.dtype)

    def cal_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C)
        return feat_mean, feat_std


    def forward(self, feats, ins, flag):
        ins = ins.squeeze() # c, feat_dim
        mean_0, std_0 = self.cal_mean_std(feats[0])
        mean_1, std_1 = self.cal_mean_std(feats[1])
        mean_2, std_2 = self.cal_mean_std(feats[2])
        mean_3, std_3 = self.cal_mean_std(feats[3])
        mean_4, std_4 = self.cal_mean_std(feats[4])
        features_0 = torch.cat([mean_0,std_0,ins],dim=1)
        features_1 = torch.cat([mean_1,std_1,ins],dim=1)
        features_2 = torch.cat([mean_2,std_2,ins],dim=1)
        features_3 = torch.cat([mean_3,std_3,ins],dim=1)
        features_4 = torch.cat([mean_4,std_4,ins],dim=1)
        out_0 = self.fc_0(features_0)
        out_1 = self.fc_1(features_1)
        out_2 = self.fc_2(features_2)
        out_3 = self.fc_3(features_3)
        out_4 = self.fc_4(features_4)
        features_un_0 = torch.cat([mean_0,std_0],dim=1)
        features_un_1 = torch.cat([mean_1,std_1],dim=1)
        features_un_2 = torch.cat([mean_2,std_2],dim=1)
        features_un_3 = torch.cat([mean_3,std_3],dim=1)
        features_un_4 = torch.cat([mean_4,std_4],dim=1)
        out_un_0 = self.fc_un_0(features_un_0)
        out_un_1 = self.fc_un_1(features_un_1)
        out_un_2 = self.fc_un_2(features_un_2)
        out_un_3 = self.fc_un_3(features_un_3)
        out_un_4 = self.fc_un_4(features_un_4)
        loss_0 = self.bce_loss(out_0, torch.ones(out_0.shape).to(self.device).to(self.dtype))
        loss_1 = self.bce_loss(out_1, torch.ones(out_1.shape).to(self.device).to(self.dtype))
        loss_2 = self.bce_loss(out_2, torch.ones(out_2.shape).to(self.device).to(self.dtype))
        loss_3 = self.bce_loss(out_3, torch.ones(out_3.shape).to(self.device).to(self.dtype))
        loss_4 = self.bce_loss(out_4, torch.ones(out_4.shape).to(self.device).to(self.dtype))
        loss_un_0 = self.bce_loss(out_un_0, torch.ones(out_un_0.shape).to(self.device).to(self.dtype))
        loss_un_1 = self.bce_loss(out_un_1, torch.ones(out_un_1.shape).to(self.device).to(self.dtype))
        loss_un_2 = self.bce_loss(out_un_2, torch.ones(out_un_2.shape).to(self.device).to(self.dtype))
        loss_un_3 = self.bce_loss(out_un_3, torch.ones(out_un_3.shape).to(self.device).to(self.dtype))
        loss_un_4 = self.bce_loss(out_un_4, torch.ones(out_un_4.shape).to(self.device).to(self.dtype))
        loss = 0.1 * (loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_un_0 + loss_un_1 + loss_un_2 + loss_un_3 + loss_un_4)

        return loss
