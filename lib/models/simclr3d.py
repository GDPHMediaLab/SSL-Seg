from matplotlib.pyplot import grid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from timm.layers.helpers import to_3tuple
from monai.networks.layers import Conv
# from networks import build_3d_sincos_position_embedding
from lib.networks.patch_embed_layers import PatchEmbed3D

__all__ = ["SimCLR3D",
           "SimCLR_ViT"]


def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)],
        dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


def build_perceptron_position_embedding(grid_size, embed_dim, num_tokens=1):
    pos_emb = torch.rand([1, np.prod(grid_size), embed_dim])
    nn.init.normal_(pos_emb, std=.02)

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    return pos_embed


def patchify_image(x, patch_size):
    """
    ATTENTION!!!!!!!
    Different from 2D version patchification: The final axis follows the order of [ph, pw, pd, c] instead of [c, ph, pw, pd]
    """
    # patchify input, [B,C,H,W,D] --> [B,C,gh,ph,gw,pw,gd,pd] --> [B,gh*gw*gd,ph*pw*pd*C]
    B, C, H, W, D = x.shape
    patch_size = to_3tuple(patch_size)
    grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

    x = x.reshape(B, C, grid_size[0], patch_size[0], grid_size[1], patch_size[1], grid_size[2],
                  patch_size[2])  # [B,C,gh,ph,gw,pw,gd,pd]
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size),
                                                  np.prod(patch_size) * C)  # [B,gh*gw*gd,ph*pw*pd*C]

    return x


def batched_shuffle_indices(batch_size, length, device):
    """
    Generate random permutations of specified length for batch_size times
    Motivated by https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    """
    rand = torch.rand(batch_size, length).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm



class SimCLR3D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 encoder,
                 decoder,
                 args):
        super().__init__()
        self.args = args
        input_size = to_3tuple(args.input_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size

        out_chans = args.in_chans * np.prod(self.patch_size)
        self.out_chans = out_chans

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        # build positional encoding for encoder and decoder
        if args.pos_embed_type == 'sincos':
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            args.encoder_embed_dim,
                                                                            num_tokens=0)
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            args.decoder_embed_dim,
                                                                            num_tokens=0)
        elif args.pos_embed_type == 'perceptron':
            self.encoder_pos_embed = build_perceptron_position_embedding(grid_size,
                                                                         args.encoder_embed_dim,
                                                                         num_tokens=0)
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            args.decoder_embed_dim,
                                                                            num_tokens=0)

        # build encoder and decoder
        from lib.networks import patch_embed_layers
        embed_layer = getattr(patch_embed_layers, args.patchembed)
        self.encoder = encoder(patch_size=patch_size,
                               in_chans=args.in_chans,
                               embed_dim=args.encoder_embed_dim,
                               depth=args.encoder_depth,
                               num_heads=args.encoder_num_heads,
                               embed_layer=embed_layer)
        self.decoder = decoder(patch_size=patch_size,
                               num_classes=out_chans,
                               embed_dim=args.decoder_embed_dim,
                               depth=args.decoder_depth,
                               num_heads=args.decoder_num_heads)

        self._build_projector_and_predictor_mlps()
    def _build_projector_and_predictor_mlps(self):
        pass

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            print("dim1:",dim1)
            print("dim2:",dim2)

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)




    def forward(self, x, return_image=False):
        args = self.args
        batch_size = x.size(0)
        in_chans = x.size(1)
        assert in_chans == args.in_chans
        out_chans = self.out_chans
        x = patchify_image(x, self.patch_size)  # [B,gh*gw*gd,ph*pw*pd*C]

        # compute length for selected and masked
        length = np.prod(self.grid_size)
        sel_length = int(length * (1 - args.mask_ratio))
       # msk_length = length - sel_length

        # generate batched shuffle indices
        shuffle_indices = batched_shuffle_indices(batch_size, length, device=x.device)
        unshuffle_indices = shuffle_indices.argsort(dim=1)

        # select and mask the input patches
        shuffled_x = x.gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))

        # select the position embedings accordingly
        all_encoder_pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1).gather(dim=1, index=unshuffle_indices[:, :,
                                                                                                      None].expand(-1,
                                                                                                                   -1,
                                                                                                                   args.encoder_embed_dim))

        # forward encoder & proj to decoder dimension
        all_x = self.encoder(shuffled_x, all_encoder_pos_embed)

        return all_x



class SimCLR_ViT(SimCLR3D):
    def _build_projector_and_predictor_mlps(self):
        hidden_dim = self.args.encoder_embed_dim
        mlp_dim = self.args.mlp_dim
        dim = self.args.output_dim
        del self.encoder.head

        # projectors
        self.encoder.head = self._build_mlp(2, hidden_dim, mlp_dim, dim)


