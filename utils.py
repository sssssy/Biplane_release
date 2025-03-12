from pdb import set_trace as debug
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pynvml


def set_global_random_seed(seed=None):
       
    GLOBAL_RANDOM_SEED = datetime.now().microsecond if seed is None else seed
    try: 
        torch.manual_seed(GLOBAL_RANDOM_SEED)
        torch.cuda.manual_seed_all(GLOBAL_RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        random.seed(GLOBAL_RANDOM_SEED)
        np.random.seed(GLOBAL_RANDOM_SEED)
        torch.set_printoptions(sci_mode=False, linewidth=os.get_terminal_size().columns)
        np.set_printoptions(suppress=True, linewidth=os.get_terminal_size().columns)
    except:
        print('[set_global_random_seed] failed.')

    return GLOBAL_RANDOM_SEED
    
    
def detect_device():
    
    ## auto detect proper device to use
    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            device = 'cpu'
        elif device_count == 1:
            device = 'cuda:0'
        else:
            pynvml.nvmlInit()
            max_free = 0
            max_free_device = 0
            for i in range(device_count):
                info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
                # print(f'total memory {i}: {info.total / 1024 / 1024} MB')
                # print(f'free {i}: {info.free / 1024 / 1024} MB')
                # print(f'used {i}: {info.used / 1024 / 1024} MB')
                if info.free > max_free:
                    max_free = info.free
                    max_free_device = i
            device = f'cuda:{max_free_device}'
            pynvml.nvmlShutdown()
    except:
        print('[detect_device] failed.')
        device='cpu'

    return device


class fc_residual_block(nn.Module):

    def __init__(self, input_units, output_units=None, last_activation=True, norm_layers=False):
        super(fc_residual_block, self).__init__()
        if not output_units:
            output_units = input_units
        middle_units = min(input_units, output_units) // 2
        if input_units != output_units:
            self.updownsample = nn.Linear(input_units, output_units)
        self.fc1 = nn.Linear(input_units, middle_units)
        self.fc2 = nn.Linear(middle_units, output_units)
        self.norm_layers = norm_layers
        if self.norm_layers:
            self.ln1 = nn.LayerNorm(middle_units)
            self.ln2 = nn.LayerNorm(output_units)
        self.relu = nn.LeakyReLU()
        self.last_activation = last_activation

    def forward(self, x):
        output = self.fc1(x)
        if self.norm_layers:
            output = self.ln1(output)
        output = self.relu(output)
        if hasattr(self, 'updownsample'):
            x = self.updownsample(x)
        output = self.fc2(output) + x
        if self.norm_layers:
            output = self.ln2(output)
        if self.last_activation:
            output = self.relu(output)
        return output


class residual(nn.Module):

    def __init__(self, *modules):
        super(residual, self).__init__()
        self.module = nn.Sequential(*modules)
        
    def forward(self, inputs):
        return self.module(inputs) + inputs

        
class trainable_layer(nn.Module):
    
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        
    def forward(self, x):
        return self.layer(x)


def fcrelu(in_dims, out_dims=None):
    return nn.Sequential(
        nn.Linear(in_dims, out_dims if out_dims is not None else in_dims),
        nn.LeakyReLU(),
    )

    
def my_print(print_func):
    from datetime import datetime
    import traceback, os
    def wrap(*args, **kwargs):
        i = -2
        call = traceback.extract_stack()[i]
        while call[2] in ('log', 'show'):
            i -= 1
            call = traceback.extract_stack()[i]
        print_func(f'\x1b[0;96;40m[{datetime.now().strftime("%H:%M:%S")} {os.path.relpath(call[0])}:{call[1]}]\x1b[0;37;49m ', end='')
        print_func(*args, **kwargs)
    return wrap
pr = print
print = my_print(print)


def check(*args):
    for x in args:
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print('[Trainer: check] nan/inf encountered.')
            debug()


def normalize(vec): ## size: [..., 3]
    return torch.nn.functional.normalize(vec, dim=-1)


def positional_encoding(wiwo, level):
    '''
    generate sin_cos positional encoding for input tensor of size [batch_size, sample_count, dims]

    Args:
        wiwo (tensor): input tensor
        level (int): pe level

    Returns:
        tensor: size [bs, sample_count, dims * level * 2]
    '''
    if level == 0:
        return wiwo
    dims = wiwo.shape[-1]
    wiwo = wiwo * np.pi
    exps = torch.tensor([2**i for i in range(level)]).tile((dims*2,))
    exps = exps.reshape(1, 1, -1).expand(wiwo.shape[0], wiwo.shape[1], -1).to(wiwo.device) ## size: [bs, qs, l*2d]
    pe = torch.repeat_interleave(wiwo, level, dim=-1).tile((1, 1, 2)) ## size: [bs, qs, d*l*2]
    pe = pe * exps
    pe[:, :, :level*dims] = torch.sin(pe[:, :, :level*dims])
    pe[:, :, level*dims:] = torch.cos(pe[:, :, level*dims:])
    return pe


def dot(x, y):
    return x[..., 0:1] * y[..., 0:1] + x[..., 1:2] * y[..., 1:2] + x[..., 2:3] * y[..., 2:3]


def cross(a, b):
    return torch.cross(a, b, dim=-1)


def coordinateSystem(axis_a):
    nxy_max, nxy_argmax = axis_a[..., 0:2].max(-1, keepdims=True)
    nz = axis_a[..., 2:3]
    zeros = torch.zeros_like(nz).to(axis_a.device)
    invLen = 1.0 / torch.sqrt(torch.clamp(nxy_max**2 + nz**2, min=0))
    axis_c0 = torch.cat([nz, zeros, -nxy_max], dim=-1) * invLen
    axis_c1 = torch.cat([zeros, nz, -nxy_max], dim=-1) * invLen
    axis_c = torch.stack([axis_c0, axis_c1], dim=0).gather(dim=0, index=nxy_argmax.expand(*nxy_argmax.shape[:-1], 3).unsqueeze(0)).squeeze(0)
    axis_b = cross(axis_c, axis_a)
    return axis_a, axis_b, axis_c
    

def eval_SG(wiwo, axis, sharpness, amp):
    '''
    evaluate Spherical Gaussian

    Args:
        wiwo ([bs, qs, 6]): wix, wiy, wox, woy, wiz, woz. Note: only wo is used here.
        axis ([bs, qs, 3]): axisx, axisy, axisz, a normalized vector
        sharpness ([bs, qs, 1]): 1 / roughness
        amp ([bs, qs, 1]): amptitude
    
    Return:
        sg ([bs, qs, 1]): SG value
    '''
    wox, woy, woz = wiwo[:, :, 2:3], wiwo[:, :, 3:4], wiwo[:, :, 5:6]
    axisx, axisy = axis[:, :, 0:1], axis[:, :, 1:2]
    axisz = torch.sqrt(torch.clamp(1 - axisx ** 2 - axisy ** 2, min=0))
    cos_wo = wox * axisx + woy * axisy + woz * axisz
    sg = amp * torch.exp(sharpness * (cos_wo - 1.0))
    return sg


def eval_ASG(wiwo, x, y, z, sg_params, eps=1e-7):
    '''
    evaluate Anisotropic Spherical Gaussian
    all input params are (0, 1)

    Args:
        wiwo ([bs, qs, 6]): wix, wiy, wox, woy, wiz, woz. Note: only wo is used here.
        x ([bs, qs, 3]): normalized Frame vector x
        y ([bs, qs, 3]): normalized Frame vector y
        z ([bs, qs, 3]): normalized Frame vector z
        sg_params:
            r1 ([bs, qs, 1]): roughness (variance 1 / lambda)
            r2 ([bs, qs, 1]): roughness (variance 1 / mu)
            v ([bs, qs, 1]): param (variance nu). Note: should be less than both lambda and mu. The bigger, the dimmer.
            amp ([bs, qs, 1]): amptitude c

    Returns:
        sg ([bs, qs, 1]): smoothed ASG value
    '''
    wox, woy, woz = wiwo[..., 2:3], wiwo[..., 3:4], wiwo[..., 5:6]
    r1, r2, v, amp = sg_params[..., 0:1], sg_params[..., 1:2], sg_params[..., 2:3], sg_params[..., 3:4] 
    cosx = wox * x[..., 0:1] + woy * x[..., 1:2] + woz * x[..., 2:3]
    cosy = wox * y[..., 0:1] + woy * y[..., 1:2] + woz * y[..., 2:3]
    cosz = wox * z[..., 0:1] + woy * z[..., 1:2] + woz * z[..., 2:3]
    exp = cosx**2 / (r1 + eps) + cosy**2 / (r2 + eps) + cosz**2 * (v / (torch.maximum(r1, r2) + eps))
    sg = amp * torch.exp(-exp)
    smooth = torch.clamp(wox * z[..., 0:1] + woy * z[..., 1:2] + woz * z[..., 2:3], min=0)
    return sg * smooth


def tonemap(x):
    return x / (1+x)


def rgb2hsv(input, epsilon=1e-10):
    '''
    convert rgb tensor into hsv

    Args:
        input (tensor, [bs, 3, ...])
        epsilon (float, optional): Defaults to 1e-10.

    Returns:
        # tensor, [bs, 4, ...]: hsv, H is encoded by sin and cos
        tensor, [bs, 3, ...]: hsv, H is in radian
    '''
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h / 180 * 3.14, s, v), dim=1)
    # return torch.stack((torch.sin(h / 180 * 3.14), torch.cos(h / 180 * 3.14), s, v), dim=1)


def rgb2hsl(input, epsilon=1e-10):
    '''
    convert rgb tensor into hsl

    Args:
        input (tensor, [bs, 3, ...])
        epsilon (float, optional): Defaults to 1e-10.

    Returns:
        # tensor, [bs, 4, ...]: hsl, H is encoded by sin and cos
        tensor, [bs, 3, ...]: hsl, H is in radian
    '''
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    l = (max_rgb + min_rgb) / 2
    s = torch.where(l < 0.5, max_min / (max_rgb + min_rgb + epsilon), max_min / (2 - max_rgb - min_rgb + epsilon))

    return torch.stack((h / 180 * 3.14, s, l), dim=1)
    # return torch.stack((torch.sin(h / 180 * 3.14), torch.cos(h / 180 * 3.14), s, l), dim=1)


def time_change(time_input):
    time_list = []
    if time_input/3600 > 1:
        time_h = int(time_input/3600)
        time_m = int((time_input-time_h*3600) / 60)
        time_s = int(time_input - time_h * 3600 - time_m * 60)
        time_list.append(str(time_h))
        time_list.append('h ')
        time_list.append(str(time_m))
        time_list.append('m ')

    elif time_input/60 > 1:
        time_m = int(time_input/60)
        time_s = int(time_input - time_m * 60)
        time_list.append(str(time_m))
        time_list.append('m ')
    else:
        time_s = int(time_input)

    time_list.append(str(time_s))
    time_list.append('s')
    time_str = ''.join(time_list)
    return time_str


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def ubo2014_angles_to_xyz(vec):
    vec = vec * np.pi / 180.0
    z = torch.sin(vec[..., 1:2])
    x = torch.cos(vec[..., 1:2]) * torch.cos(vec[..., 0:1])
    y = torch.cos(vec[..., 1:2]) * torch.sin(vec[..., 0:1])
    return normalize(torch.cat([x, y, z], dim=-1))


def xy_to_xyz(vec):
    return torch.cat([vec[..., 0:1], vec[..., 1:2], torch.sqrt(torch.clamp(1 - vec[..., 0:1] ** 2 - vec[..., 1:2] ** 2, 0, 1))], dim=-1)
    

def xyz_to_thetaphi(vec):
    vec = normalize(vec)
    # vec[vec >= 1] = 0.999999 ## to avoid nan in arccos (caused by float accuracy)
    theta = torch.arccos(vec[..., 2:3])
    phi = torch.atan2(vec[..., 1:2], vec[..., 0:1])
    return torch.cat([theta, phi], dim=-1) ## [0. 0.5pi] [-pi, pi]


def thetaphi_to_xyz(vec):
    z = torch.cos(vec[..., 0:1])
    x = torch.sin(vec[..., 0:1]) * torch.cos(vec[..., 1:2])
    y = torch.sin(vec[..., 0:1]) * torch.sin(vec[..., 1:2])
    return normalize(torch.cat([x, y, z], dim=-1))


def rotate_along_axis(vec, axis, angle):
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    out = vec * cos_angle
    out += axis * (axis[..., 0:1] * vec[..., 0:1] + axis[..., 1:2] * vec[..., 1:2] + axis[..., 2:3] * vec[..., 2:3]) * (1 - cos_angle)
    out += torch.cross(axis.expand_as(vec), vec) * sin_angle
    return out


def wiwo_xyz_to_hd_thetaphi(wi, wo):

    ## compute half
    h = (wi + wo) / 2
    h = normalize(h)
    h = xyz_to_thetaphi(h)

    ## compute diff
    n = torch.tensor([0, 0, 1], dtype=wi.dtype, device=wi.device)
    bi_n = torch.tensor([0, 1, 0], dtype=wi.dtype, device=wi.device)
    ## here inputs h_theta/phi and output d_xyz (normalized)
    d = rotate_along_axis(rotate_along_axis(wi, n, -h[..., 1:2]), bi_n, -h[..., 0:1]) 
    d = xyz_to_thetaphi(d)
    
    return h, d
    

class Frame:
    
    def __init__(self, n, t=None):
        #! assume v is normalized
        # n = normalize(n)
        # self.normal, self.bitangent, self.tangent = coordinateSystem(n)
        self.normal = n
        if t == None:
            # assume v[2] is non-zero (typically positive)
            self.bitangent = normalize(torch.cat([n[..., 2:3], torch.zeros_like(n[..., 2:3]), -n[..., 0:1]], dim=-1))
        else:
            self.bitangent = normalize(cross(n, t))
        self.tangent = cross(n, self.bitangent)
        
    def to_world(self, v):
        return self.tangent * v[..., 0:1] + self.bitangent * v[..., 1:2] + self.normal * v[..., 2:3]
    
    def to_local(self, v):
        return torch.cat([dot(self.tangent, v), dot(self.bitangent, v), dot(self.normal, v)], dim=-1)