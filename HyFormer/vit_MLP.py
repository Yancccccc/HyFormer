##没有CA 只有MLP
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.functional as F

class MCA(nn.Module):
    def __init__(self, dim, n_heads, attn_drp=0., drp=0.1):
        super(MCA, self).__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drp = nn.Dropout(p=attn_drp)
        self.proj = nn.Linear(dim, dim)
        self.block_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.block_drop = nn.Dropout(p=drp)
    def forward(self, x):
        shape = x.shape[: -1] + (3, self.n_heads, self.head_dim)
        qkv = self.qkv(self.block_norm1(x)).view(shape)  # batch_size, n_patches , 3, n_heads, heads_dim
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, batch_size, n_heads, n_patches , heads_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (batch_size, n_heads, n_patches , heads_dim)
        attn = q.transpose(-1, -2) @ k * self.scale
        attn = self.attn_drp(attn.softmax(dim=-1))
        attn_out = attn @ v.transpose(-1, -2)  # (batch_size, n_heads, n_patches , head_dim)
        attn_out = attn_out.permute(0, 3, 2, 1)
        attn_out = attn_out.flatten(2)
        att = self.block_drop(self.proj(attn_out))  # (batch_size, n_patches + 1, dim)
        return att

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
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads
        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
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
        out = self.to_out(out)               ######att
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))
    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn,ff in self.layers:     # 与self.layers.append(nn.ModuleList）中的顺序要对应
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x

class ViT_MLP(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=4, dim_head = 16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()

        patch_dim = image_size ** 2 * near_band
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        self.action = nn.GELU()
        self.mlp_norm = nn.LayerNorm(dim)
        #self.feat_spe = nn.Linear(channels*near_band, dim)
        self.feat_ss = nn.Linear(dim*2, dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.feat_spe = nn.Linear(8, dim)  ##input_dim= channel * band_patchs

    def forward(self, x, mask = None):
       
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x1 = self.patch_to_embedding(x) #[b,n,dim]
        b, n, _ = x1.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) #[b,1,dim]
        x1 = torch.cat((cls_tokens, x1), dim = 1) #[b,n+1,dim]
        x1 += self.pos_embedding[:, :(n + 1)]
        x1 = self.dropout(x1)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x1 = self.transformer(x1, mask)

        # classification: using cls_token output
        x1 = self.to_latent(x1[:,0])  #

        x2 = torch.flatten(x,start_dim=1, end_dim=2)
        x2 = self.feat_spe(x2)
        x2 = self.action(x2)
        x3 = torch.cat([x1, x2], dim=1)
        x3 = self.feat_ss(x3)
        x3 = self.action(x3)
        #x3 = self.action(x3)
        x3 = self.mlp_norm(x3)
        # MLP classification layer
        return self.classifier(x3)
