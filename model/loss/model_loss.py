import torch
import torch.nn.functional as F

def reconstruction_loss(x, x_hat):
    # 方程 (6): L_recon = E[(x - x_hat)^2]
    return F.mse_loss(x, x_hat)

def alignment_loss(g_s_global, clip_features):
    # 方程 (7): L_CLIP = E[||G_s_global - CLIP(x)||^2]
    return F.mse_loss(g_s_global, clip_features)

def mmd_loss(phi_c, phi_g_s_global, phi_g):
    # 方程 (9): L_MMD^2
    term_1 = torch.mean((phi_c - phi_g_s_global) ** 2)
    term_2 = torch.mean((phi_g_s_global - phi_g) ** 2)
    term_3 = torch.mean((phi_c - phi_g) ** 2)
    return term_1 + term_2 - 2 * term_3

def alignment_loss_combined(l_clip, l_mmd, beta):
    # 方程 (8): L_align = beta * L_CLIP + (1 - beta) * L_MMD
    return beta * l_clip + (1 - beta) * l_mmd

def total_loss(l_recon, l_align, lambda_):
    # 方程 (9): L_train = lambda * L_recon + (1 - lambda) * L_align
    return lambda_ * l_recon + (1 - lambda_) * l_align

def discriminator_loss(d_real, d_fake):
    # 方程 (11): L_disc = E[log D(c)] - E[log(1 - D(G(S)))]
    # 计算判别器对真实样本的损失和对生成样本的损失
    # loss_real 代表判别器对真实样本的输出应该接近 1，因此取负的对数期望值
    loss_real = -torch.mean(torch.log(d_real + 1e-8))
    # loss_fake 代表判别器对生成样本的输出应该接近 0，因此取 1 减去 d_fake，再取负的对数期望值
    loss_fake = -torch.mean(torch.log(1 - d_fake + 1e-8))
    # 最终的判别器损失是两者之和
    return loss_real + loss_fake