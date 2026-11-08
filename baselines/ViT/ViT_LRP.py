""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
from einops import rearrange
from modules.layers_ours import *
from functools import partial
import sys

from baselines.ViT.helpers import load_pretrained
from baselines.ViT.weight_init import trunc_normal_
from baselines.ViT.layer_helpers import to_2tuple

apvit_pth = "/home/projects/bagon/ilanaveh/code/Transformers/APViT"
sys.path.insert(0, apvit_pth)
from mmcls.models.utils import top_pool
from mmcls.models.vit.layers import DropPath


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


def compute_rollout_attention_pooled(
        all_layer_matrices,
        keep_indices=None,
        start_layer=0):
    # Based on compute_rollout_attention, but allow for layers with different token counts (due to APViT pooling): add
    # residual separately for every layer.
    matrices = []

    for matrix in all_layer_matrices:

        B, N, _ = matrix.shape

        eye = torch.eye(
            N,
            device=matrix.device,
            dtype=matrix.dtype
        ).unsqueeze(0).expand(B, N, N)

        matrices.append(matrix + eye)

    joint_attention = matrices[start_layer]

    for i in range(start_layer + 1, len(matrices)):

        # Pooling takes place AFTER attention in block i-1,
        # and BEFORE attention in block i.
        if keep_indices is not None:

            keep = keep_indices[i - 1]

            if keep is not None:

                # Accept [B,K,C], [B,K,1], or [B,K]
                if keep.ndim == 3:
                    keep = keep[:, :, 0]

                keep = keep.long()

                # joint_attention:
                # [B, N_previous, N_start]
                #
                # Select the rows surviving pooling:
                # -> [B, K, N_start]

                row_index = keep.unsqueeze(-1).expand(
                    -1,
                    -1,
                    joint_attention.shape[-1]
                )

                joint_attention = joint_attention.gather(
                    dim=1,
                    index=row_index
                )

        # [B, N_current, N_current]
        #       @
        # [B, N_current, N_start]
        #
        # ->
        # [B, N_current, N_start]
        joint_attention = matrices[i].bmm(
            joint_attention
        )

    return joint_attention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)


class PoolingAttention(Attention):
    """
    Inherit Attention defined here (with LRP relprop function), but add code from
    APViT/mmcls/models/backbones/modules/vit_pooling.py>PoolingAttention, for pooling functionality.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., pool_config=None):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.dim = dim
        self.pool_config = pool_config

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        self.save_v(v)

        attn = self.matmul1([q, k]) * self.scale  # [B, head_num, token_num, token_num]

        if self.pool_config:
            attn_method = self.pool_config.get('attn_method')
            if attn_method == 'SUM_ABS_1':
                attn_weight = attn[:, :, 0, :].transpose(-1, -2)  # [B, token_num, head_num]
                attn_weight = torch.sum(torch.abs(attn_weight), dim=-1).unsqueeze(-1)
            elif attn_method == 'SUM':
                attn_weight = attn[:, :, 0, :].transpose(-1, -2)  # [B, token_num, head_num]
                attn_weight = torch.sum(attn_weight, dim=-1).unsqueeze(-1)
            elif attn_method == 'MAX':
                attn_weight = attn[:, :, 0, :].transpose(-1, -2)
                attn_weight = torch.max(attn_weight, dim=-1)[0].unsqueeze(-1)
            else:
                raise ValueError('Invalid attn_method: %s' % attn_method)

            # attn_weight = torch.rand(attn_weight.shape, device=attn_weight.device)
            keep_index = top_pool(attn_weight, dim=self.dim, **self.pool_config)    # [B, N, d]
        else:
            keep_index = None

        # IN 26/1/26: Add for visualizing attention (viz_attention_vit.py)
        self.attn_weight = attn_weight
        self.keep_index = None if keep_index is None else keep_index[:, :, 0]  # [B, N]

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        x = self.matmul2([attn, v]).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, keep_index

        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #
        # attn = self.matmul1([q, k])  # [B, head_num, token_num, token_num]
        #
        # if self.pool_config:
        #     attn_method = self.pool_config.get('attn_method')
        #     if attn_method == 'SUM_ABS_1':
        #         attn_weight = attn[:, :, 0, :].transpose(-1, -2)  # [B, token_num, head_num]
        #         attn_weight = torch.sum(torch.abs(attn_weight), dim=-1).unsqueeze(-1)
        #     elif attn_method == 'SUM':
        #         attn_weight = attn[:, :, 0, :].transpose(-1, -2)  # [B, token_num, head_num]
        #         attn_weight = torch.sum(attn_weight, dim=-1).unsqueeze(-1)
        #     elif attn_method == 'MAX':
        #         attn_weight = attn[:, :, 0, :].transpose(-1, -2)
        #         attn_weight = torch.max(attn_weight, dim=-1)[0].unsqueeze(-1)
        #     else:
        #         raise ValueError('Invalid attn_method: %s' % attn_method)
        #
        #     # attn_weight = torch.rand(attn_weight.shape, device=attn_weight.device)
        #     keep_index = top_pool(attn_weight, dim=self.dim, **self.pool_config)
        # else:
        #     keep_index = None
        #
        # # IN 26/1/26: Add for visualizing attention (viz_attention_vit.py)
        # self.attn_weight = attn_weight
        # if keep_index is not None:
        #     self.keep_index = top_pool(attn_weight, dim=1, **self.pool_config)  # dim=1
        # else:
        #     self.keep_index = None
        #
        # attn = self.softmax(attn, dim=-1)
        # attn = self.attn_drop(attn)
        #
        # attn.register_hook(self.save_attn_gradients)
        #
        # x = self.matmul2([attn, v]).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x, keep_index


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PoolingBlock(Block):
    """
    Inherit Block defined here (with LRP relprop function), but add code from
    APViT/mmcls/models/backbones/modules/vit_pooling.py>PoolingBlock, for pooling functionality.
    Create new relprop function, for reversing token-pooling before back-propagation.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 pool_config=None):
        # Removed drop_path from original PoolingBlock since it's 0 anyway.
        super().__init__(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         drop=drop, attn_drop=attn_drop)
        self.attn = PoolingAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                     proj_drop=drop, pool_config=pool_config)

        # Information needed to reverse the token selection (for relprop)
        self.pool_keep_index = None
        self.num_tokens_before_pool = None

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        feature, keep_index = self.attn(self.norm1(x2))
        x = self.add1([x1, feature])

        # Save information needed by relprop
        self.pool_keep_index = None
        self.num_tokens_before_pool = x.shape[1]

        if keep_index is not None:
            if keep_index.shape[1] != x.shape[1]:
                self.pool_keep_index = keep_index

                # ToDo: should there be a new gather function, with relprop?
                x = x.gather(dim=1, index=keep_index)

        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):

        # --------------------------------------------------
        # Reverse the MLP residual branch
        # --------------------------------------------------

        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        # --------------------------------------------------
        # REVERSE TOKEN POOLING
        # K tokens -> N tokens
        # --------------------------------------------------

        if self.pool_keep_index is not None:
            full_cam = cam.new_zeros(
                cam.shape[0],
                self.num_tokens_before_pool,
                cam.shape[2]
            )

            full_cam.scatter_add_(
                dim=1,
                index=self.pool_keep_index,
                src=cam
            )

            cam = full_cam

        # --------------------------------------------------
        # Reverse the attention residual branch
        # --------------------------------------------------

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)

        return cam


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1,2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        self.norm = LayerNorm(embed_dim)
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head = Linear(embed_dim, num_classes)

        # FIXME not quite sure what the proper weight init is supposed to be,
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            keep_indices = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
                # Pooling performed AFTER this block's attention
                keep_indices.append(getattr(blk, "pool_keep_index", None))
            if isinstance(self, APViT_LRP):
                rollout = compute_rollout_attention_pooled(cams, keep_indices=keep_indices, start_layer=start_layer)
                cam = rollout[:, 0, 1:]
                if self.cnn_keep_indexes is not None:
                    full_cam = cam.new_zeros(cam.shape[0], self.cnn_num_tokens_before_pool)
                    token_ids = self.get_original_patch_indices(start_layer=start_layer, device=cam.device)  # for dealing with start_layer!=1.
                    patch_ids = token_ids[:, 1:]
                    full_cam.scatter_add_(dim=1, index=patch_ids, src=cam)
                    cam = full_cam
            else:
                rollout = compute_rollout_attention(cams, start_layer=start_layer)
                cam = rollout[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


class APViT_LRP(VisionTransformer):
    """
    Inherit VisionTransformer defined here (with LRP mechanisms), but add code from
    APViT/mmcls/models/vit/vit_siam_merge.py>PoolingViT, for Pooling functionality.
    """
    def __init__(self, embed_dim, in_channels,
                 attn_method='SUM_ABS_1',
                 cnn_pool_config=None,
                 vit_pool_configs=None,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 **kwargs):
        super().__init__(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, **kwargs)
        num_patches = self.pos_embed.shape[1] - 1
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.projs = nn.ModuleList([nn.Conv2d(in_channels, embed_dim, 1,)])
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config

        self.cnn_keep_indexes = None
        self.cnn_num_tokens_before_pool = None

        self.patch_add = Add()
        self.cls_add = Add()

        if attn_method == 'SUM':
            self.attn_f = lambda x: torch.sum(x, dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_1':
            self.attn_f = lambda x: torch.sum(torch.abs(x), dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_2':
            self.attn_f = lambda x: torch.sum(torch.pow(torch.abs(x), 2), dim=1).unsqueeze(1)
        elif attn_method == 'MAX':
            self.attn_f = lambda x: torch.max(x, dim=1)[0].unsqueeze(1)
        elif attn_method == 'MAX_ABS_1':
            self.attn_f = lambda x: torch.max(torch.abs(x), dim=1)[0].unsqueeze(1)
        elif attn_method == 'Random':
            self.attn_f = lambda x: x[:, torch.randint(high=x.shape[1], size=(1,))[0], ...].unsqueeze(1)
        else:
            raise ValueError("Unknown attn_method")

        if vit_pool_configs is not None:
            # If  vit_pool_configs is None, I used blocks defined here, even though in original PoolingViT, they used
            # their own block. I think they should be equivalent.
            vit_keep_rates = vit_pool_configs['keep_rates']
            self.blocks = nn.ModuleList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    pool_config=dict(keep_rate=vit_keep_rates[i], **vit_pool_configs),
                )
                for i in range(depth)])

    def forward(self, x):
        x = x[0]

        # x = self.patch_embed(x)
        x = self.projs[0](x)

        attn_map = self.attn_f(x)  # [B, 1, H, W]
        attn_weight = attn_map.flatten(2).transpose(2, 1)

        x = x.flatten(2).transpose(1, 2)  # [B, D, H, W] -> [B, H*W, D]

        B, N, C = x.shape

        x = self.patch_add([x, self.patch_pos_embed])

        self.cnn_num_tokens_before_pool = x.shape[1]
        self.cnn_keep_indexes = None

        if self.cnn_pool_config is not None:
            keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)

            if keep_indexes is not None:
                self.cnn_keep_indexes = keep_indexes[:, :, 0].long()
                x = x.gather(dim=1, index=keep_indexes)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_add([cls_tokens, self.cls_pos_embed])

        x = torch.cat((cls_tokens, x), dim=1)
        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def get_original_patch_indices(self, start_layer, device):
        """
        Return the original CNN-patch index corresponding to each token
        entering Transformer block `start_layer`.

        CLS is represented by -1.
        """

        if self.cnn_keep_indexes is not None:
            patch_ids = self.cnn_keep_indexes
        else:
            # No initial CNN pooling
            B = 1  # current attribution code assumes batch size 1
            patch_ids = torch.arange(
                self.cnn_num_tokens_before_pool,
                device=device
            ).unsqueeze(0).expand(B, -1)

        patch_ids = patch_ids.to(device)

        # First Transformer block receives:
        # [CLS, selected patch 1, selected patch 2, ...]
        cls_id = torch.full(
            (patch_ids.shape[0], 1),
            -1,
            dtype=torch.long,
            device=device
        )

        token_ids = torch.cat(
            [cls_id, patch_ids],
            dim=1
        )

        # Reproduce all ViT pruning that occurred BEFORE start_layer
        for i in range(start_layer):
            keep = getattr(self.blocks[i], 'keep_index', None)

            if keep is not None:
                if keep.ndim == 3:
                    keep = keep[:, :, 0]

                token_ids = token_ids.gather(
                    dim=1,
                    index=keep.long()
                )

        return token_ids


    @property
    def no_weight_decay(self):
        return {'patch_pos_embed', 'cls_pos_embed', 'cls_token'}


def deit_base_patch16_224_apvit(pretrained=False, use_pooling=True, **kwargs):

    pooling_args = {'attn_method': 'SUM_ABS_1',
                    'cnn_pool_config': dict(keep_num=160, exclude_first=False) if use_pooling else None,
                    'vit_pool_configs': dict(keep_rates=[1.] * 4 + [0.9] * 8, exclude_first=True, attn_method='SUM') if use_pooling else None}
    # pooling_args={}
    model = APViT_LRP(
        in_channels=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        **pooling_args, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
