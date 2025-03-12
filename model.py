
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils import fc_residual_block, fcrelu, Frame, xy_to_xyz, normalize
from config import RepConfig

class Decoder(nn.Module):

    def __init__(self, config: RepConfig):
        super().__init__()
        self.network_name = type(self).__name__
        self.config = config
        self.init_decoder()

    def get_param_count(self):
        return f'{sum(param.numel() for param in self.decoder.parameters()):,d}'

    def init_decoder(self):
        channels = 256
        self.decoder = nn.Sequential(
            fcrelu(self.config.query_size + sum(self.config.latent_size[:2]), channels),
            fc_residual_block(channels, channels),
            fc_residual_block(channels, channels),
            nn.Linear(channels, 3),
            nn.Sigmoid(),
        )

    def forward(self, x): ## x size: [bs, query_per_step, wiwo + latent_size]
        return self.decoder(x)


class Adapter(nn.Module):
    
    def __init__(self, config: RepConfig):
        super().__init__()
        self.network_name = type(self).__name__
        self.config = config
        self.U, self.V = config.btf_size[:2]
        self.M = len(config.train_materials)
        self.m = nn.Parameter(0.3333 * torch.ones(
            [1, 1, 3, 3], dtype=torch.float32, device=self.config.device, requires_grad=True
        ).repeat(self.M, self.U * self.V, 1, 1))
        self.b = nn.Parameter(torch.zeros([1, 1, 1, 3], dtype=torch.float32, device=self.config.device, requires_grad=True).repeat(self.M, self.U * self.V, 1, 1))

    def get_param_count(self):
        return f'{self.M} x {self.U} x {self.V} x 12 = {self.M * self.U * self.V * 12:,d}'
    
    def vis_feature(self, out_dir, epoch):
        from feature_vis import write_blocks
        import os
        write_blocks(self.m.detach().cpu().numpy().reshape(self.M, self.U, self.V, 9), os.path.join(out_dir, f'epoch-{epoch}_feature_vis', 'adapter_m.exr'))
        write_blocks(self.b.detach().cpu().numpy().reshape(self.M, self.U, self.V, 3), os.path.join(out_dir, f'epoch-{epoch}_feature_vis', 'adapter_b.exr'))
    
    def get_uv_indices(self, u, v): ## uv in [0, 1]
        ind_u = u * self.U
        ind_v = v * self.V
        ind_u[ind_u == self.U] = self.U - 1
        ind_v[ind_v == self.V] = self.V - 1
        return ind_u, ind_v ## float

    def bilinear_interp(self, x, m, i, j, I, J):
        '''
            adapter's blur() and bilinear_interp() firstly reshape input into (M, U, V, L), which is the same as that in DualTriPlane
        '''
        x = x.reshape(self.M, self.U, self.V, -1)
        i1, j1 = map(lambda x: torch.floor(x).long(), (i, j))
        i2, j2 = (i1 + 1) % I, (j1 + 1) % J
        ir, jr = map(lambda x: x.float().unsqueeze(-1), (i - i1, j - j1))
        return (x[m, i1, j1, :] * (1 - ir) + x[m, i2, j1, :] * ir) * (1 - jr) + (x[m, i1, j2, :] * (1 - ir) + x[m, i2, j2, :] * ir) * jr

    def blur(self, m, radius):
        ## m: [uv, 3, 3]
        if radius < 1:
            return m
        sigma = radius / 3.0
        x_grid = np.linspace(-radius, radius, radius * 2 + 1)
        gaussian_weight = np.exp( - x_grid * x_grid / (2 * sigma ** 2) )
        gaussian_weight /= gaussian_weight.sum()
        kernel = torch.tensor(gaussian_weight).reshape(1, 1, radius * 2 + 1).float().to(self.config.device)
        
        ## do 1d-blurring on each channel
        m = m.reshape(self.M, self.U, self.V, -1)
        M, h, w, c = m.shape
        output = F.pad(m.permute(0, 3, 1, 2).reshape(-1, 1, w), mode='replicate', pad=(radius, radius))
        output = F.conv1d(output, kernel).reshape(M, c, h, w)
        output = F.pad(output.permute(0, 1, 3, 2).reshape(-1, 1, h), mode='replicate', pad=(radius, radius))
        output = F.conv1d(output, kernel).reshape(M, c, w, h).permute(0, 3, 2, 1)
        return output # (M, h, w, c)

    def forward(self, x, m, u, v, radius=0):
        # x: [bs, 3] or [bs, 1]
        ind_u, ind_v = self.get_uv_indices(u, v)
        out = torch.bmm(x.reshape(-1, 1, 3), self.bilinear_interp(self.blur(self.m, radius=radius), m, ind_u, ind_v, self.U, self.V).reshape(-1, 3, 3))
        out = out + self.bilinear_interp(self.blur(self.b, radius=radius), m, ind_u, ind_v, self.U, self.V).reshape(-1, 1, 3)
        return out.reshape(-1, 3)


class DualBiPlane(nn.Module):
    
    def __init__(self, config: RepConfig, init='zeros'):
        super().__init__()
        self.network_name = type(self).__name__
        self.config = config
        self.M = len(config.train_materials)
        self.Hx = self.Hy = config.decom_H_reso
        self.H = self.Hx * self.Hy
        self.U, self.V = config.btf_size
        self.Lxy, self.Luv = self.config.latent_size[:2]
        self.L = sum(self.config.latent_size[:2])
        if self.config.compress_only and self.config.use_hxy_comb:
            self.train_Fxys = F.interpolate(self.load_checkpoint_Fxy().permute(0, 3, 1, 2), (config.decom_H_reso, config.decom_H_reso)).permute(0, 2, 3, 1)
            self.comb_weights = nn.Parameter(getattr(torch, init)(self.M, self.train_Fxys.shape[0], 1, 1, 1, requires_grad=True, device=config.device, dtype=torch.float32))
        else:
            self.Fxy = nn.Parameter(getattr(torch, init)(self.M, self.Hx, self.Hy, self.Lxy, requires_grad=True, device=config.device, dtype=torch.float32))
        self.Fuv = nn.Parameter(getattr(torch, init)(self.M, self.U,  self.V,  self.Luv, requires_grad=True, device=config.device, dtype=torch.float32))
        
    def get_param_count(self):
        return (f'{self.M} x ({self.Hx} x {self.Hy} + {self.U} x {self.V}) x {self.L} = '
                f'{self.M * (self.Hx * self.Hy + self.U * self.V) * (self.L):,d} ')

    def load_checkpoint_Fxy(self):
        import os
        save_dict = torch.load(os.path.join(self.config.root, self.config.save_root, self.config.checkpoint_path))
        if '.state_dict' in self.config.checkpoint_path:
            raise NotImplementedError()
        else:
            Fxy = save_dict['decom'].Fxy
        Fxy.requires_grad = False
        return Fxy
    
    def vis_feature(self, out_dir, epoch):
        from feature_vis import write_blocks
        import os
        write_blocks(self.Fxy.detach().cpu().numpy(), os.path.join(out_dir, f'epoch-{epoch}_feature_vis', 'Fxy.exr'))
        write_blocks(self.Fuv.detach().cpu().numpy(), os.path.join(out_dir, f'epoch-{epoch}_feature_vis', 'Fuv.exr'))
        if self.config.compress_only and self.config.use_hxy_comb:
            np.savetxt(os.path.join(out_dir, f'epoch-{epoch}_feature_vis', 'Fxy_comb.txt'), self.comb_weights.detach().cpu().numpy().reshape(-1, 1), fmt="%.4f")
            for i, mat in enumerate(self.config.train_materials):
                heatmap = self.comb_weights[i].expand_as(self.train_Fxys[..., :3]).detach().cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
                weights = heatmap[:, 0, 0, 0].reshape(-1)
                for j, w in enumerate(weights):
                    heatmap[j, int(w*(self.config.decom_H_reso - 1)):] = 0.0
                write_blocks(np.concatenate([self.train_Fxys.detach().cpu().numpy(), heatmap], -1), os.path.join(out_dir, f'epoch-{epoch}_feature_vis', f'Fxy_comb_{mat}.exr'))
    
    def get_hxy_indices(self, h):
        ind_hx = (h[..., 0] + 1) / 2 * self.Hx
        ind_hy = (h[..., 1] + 1) / 2 * self.Hy
        ind_hx[ind_hx == self.Hx] = self.Hx - 1
        ind_hy[ind_hy == self.Hy] = self.Hy - 1
        return ind_hx, ind_hy ## float        

    def get_uv_indices(self, u, v): ## uv in [0, 1]
        ind_u = u * self.U
        ind_v = v * self.V
        ind_u[ind_u == self.U] = self.U - 1
        ind_v[ind_v == self.V] = self.V - 1
        return ind_u, ind_v ## float

    def bilinear_interp(self, x, m, i, j, I, J):
        i1, j1 = map(lambda x: torch.floor(x).long(), (i, j))
        i2, j2 = (i1 + 1) % I, (j1 + 1) % J
        ir, jr = map(lambda x: x.float().unsqueeze(-1), (i - i1, j - j1))
        return (x[m, i1, j1, :] * (1 - ir) + x[m, i2, j1, :] * ir) * (1 - jr) + (x[m, i1, j2, :] * (1 - ir) + x[m, i2, j2, :] * ir) * jr
    
    def blur(self, m, radius, use_avg=False):
        ## m: [Hx, Hy, U, V, L]
        if radius < 1:
            return m
        sigma = radius / 3.0
        x_grid = np.linspace(-radius, radius, radius * 2 + 1)
        gaussian_weight = np.exp( - x_grid * x_grid / (2 * sigma ** 2) )
        if use_avg:
            gaussian_weight = np.ones_like(gaussian_weight)
        gaussian_weight /= gaussian_weight.sum()
        kernel = torch.tensor(gaussian_weight).reshape(1, 1, radius * 2 + 1).float().to(self.config.device)
        
        ## do 1d-blurring on each channel
        M, h, w, c = m.shape
        output = F.pad(m.permute(0, 3, 1, 2).reshape(-1, 1, w), mode='replicate', pad=(radius, radius))
        output = F.conv1d(output, kernel).reshape(M, c, h, w)
        output = F.pad(output.permute(0, 1, 3, 2).reshape(-1, 1, h), mode='replicate', pad=(radius, radius))
        output = F.conv1d(output, kernel).reshape(M, c, w, h).permute(0, 3, 2, 1)
        return output
    
    def forward(self, m, h, u, v, radius=0):
        ## h: [bs, 2], u: [bs, 1], v: [bs, 1]\
        ind_hx, ind_hy = self.get_hxy_indices(h)
        ind_u, ind_v = self.get_uv_indices(u, v)
        
        if self.config.compress_only and self.config.use_hxy_comb:
            Fxy_shape = self.train_Fxys.shape ## [train_M, reso, reso, Lxy]
            self.Fxy = (self.comb_weights.expand(self.M, *Fxy_shape) * self.train_Fxys.unsqueeze(0).expand(self.M, *Fxy_shape)).sum(1)
        
        latent = torch.cat([
            self.bilinear_interp(self.blur(self.Fxy, radius=radius, use_avg=False), m, ind_hx, ind_hy, self.Hx, self.Hy),
            self.bilinear_interp(self.blur(self.Fuv, radius=radius, use_avg=False), m, ind_u,  ind_v,  self.U , self.V ),
        ], dim=-1)
        return latent


DualTriPlane = DualBiPlane