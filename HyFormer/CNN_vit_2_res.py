import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.functional as F
from torchsummary import summary
####CNN提取特征:提取3个不同层次的特征res融合为CAF



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        # for _ in range(depth - 2):
        #     self.skipcat.append(nn.Conv2d(8, 8, [1, 2], 1, 0))
        self.conv1 = nn.Conv2d(8, 8, [1, 2], 1, 0)
    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 0:
                    x = self.conv1(torch.cat([x.unsqueeze(3), last_output[nl].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1
        return x

class SE(nn.Module):
    def __init__(self, in_dim, reduction):
        super(SE, self).__init__()
        self.gload_avg_pool =  torch.nn.AdaptiveAvgPool2d(1)
        self.selayer = nn.Sequential(nn.Linear(in_dim,in_dim//reduction,bias=False),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(in_dim//reduction,in_dim),
                                     nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gload_avg_pool(x).view(b, c)
        y = self.selayer(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        return y

class FC_SE_cnn(nn.Module):
    def __init__(self, num_features=8,dim=64,num_classes=9):
        super(FC_SE_cnn, self).__init__()
        self.conv0 = nn.Conv2d(16, 64, kernel_size=1, stride=1,
                               bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.num_features = num_features
        # Append new layers
        n_fc1 = 256
        self.feat_spe = nn.Linear(self.num_features,n_fc1)
        self.classifier = nn.Linear(dim, num_classes)
        self.se = SE(in_dim=dim,reduction=4)
    def forward(self, x):
        x = torch.flatten(x,start_dim=1,end_dim=2)  # (4,2) → 8
        x = self.feat_spe(x)   # 8→256
        x = x.reshape([x.size(0),16,4,4]) # batch,16,4,4
        x = self.conv0(x)
        x_res = x
        x = self.conv1(x)
        x = self.relu(self.se(x) + x_res)
        x = self.avgpool(x)
        x_res = x
        x = self.conv2(x)
        x = self.relu(self.se(x) + x_res)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        return self.classifier(x)

class cnn_vit_2_res(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=4, dim_head = 16, dropout=0., emb_dropout=0., mode='ViT',num_features=4):
        super().__init__()
        ###############
        self.gload_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Conv2d(16, 64, kernel_size=1, stride=1,
                               bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.num_features = num_features
        # Append new layers
        n_fc1 = 256
        self.feat_spe = nn.Linear(4,256)
        self.classifier = nn.Linear(dim, num_classes)
        self.se = SE(in_dim=dim,reduction=4)
        #########################################
        patch_dim = image_size ** 2 * near_band
        self.layer1 = nn.Linear(4,64)
        self.pos_embedding = nn.Parameter(torch.randn(1, 7 + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.action = nn.GELU()
        self.mlp_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        #self.cnn = cnn(num_features=num_features)
        self.feat_ss = nn.Linear(dim * 2, dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x, mask = None):
        x6 = x.clone()
        x1 = torch.squeeze(x)
        x1 = self.layer1(x1)
        x = torch.flatten(x,start_dim=1,end_dim=2)  # (4,2) → 8
        x = self.feat_spe(x)   # 8→256
        x = x.reshape([x.size(0),16,4,4]) # batch,16,4,4
        x = self.conv0(x)
        x_res = x  #（64，4，4）
        x = self.conv1(x)
        x2 = self.relu(self.se(x) + x_res)     ##（64，4，4）
        x2_1 = self.avgpool(x2)
        x_res = x2_1
        x2_1 = self.conv2(x2_1)
        x3 = self.relu(self.se(x2_1) + x_res)   #（64，2，2）
        x3_1 = self.avgpool(x3)
        x3_1 = x3_1.view(x.size(0), -1)
        x4 = self.relu(x3_1)              #（64，1，1）
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')
        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        #x1 = self.patch_to_embedding(x) #[b,n,dim]
        x1 = torch.unsqueeze(x1,dim=1)
        x2 = torch.squeeze(self.gload_avg_pool(x2))
        x2 = torch.unsqueeze(x2,dim=1)
        x3 = torch.squeeze(self.gload_avg_pool(x3))
        x3 = torch.unsqueeze(x3, dim=1)
        x4 = torch.unsqueeze(x4,dim=1)

        x6 = self.patch_to_embedding(x6)
        b, n, _ = x6.shape
        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) #[b,1,dim]
        x5 = torch.cat((cls_tokens, x6, x2, x3, x4), dim = 1) #[b,n+1,dim]
        b1,n1,_ = x5.shape
        x5 += self.pos_embedding[:, :(n1 + 1)]
        x5 = self.dropout(x5)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x5 = self.transformer(x5, mask)

        # classification: using cls_token output
        x5 = self.to_latent(x5[:,0])  # (32,64)

        #x2 = self.cnn(x)
        #x6 = torch.cat([x5, x3_1], dim=1)
        #x6 = self.feat_ss(x6)
        #x6 = self.action(x6)
        #x3 = self.action(x3)
        #x6 = self.mlp_norm(x5)
        # MLP classification layer
        return self.mlp_head(x5)

if __name__ == '__main__':
    model = cnn_vit_2_res(
        image_size=1,
        near_band=1,
        num_patches=4,
        num_classes=9,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=8,
        dropout=0.1,
        emb_dropout=0.1,
        mode='ViT',
        channels=4,
        num_features=4
)
    model.cuda()
    summary(model, (4, 1))
