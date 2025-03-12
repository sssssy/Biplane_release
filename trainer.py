import os
from pdb import set_trace as debug
from time import perf_counter as clock
from typing import List

import tqdm
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

import model
import exr
from utils import inplace_relu, trainable_layer, optimizer_to, time_change, print
from config import RepConfig
from render import render

class BaseTrainer:

    def __init__(self, config: RepConfig, datasets: List[torch.utils.data.Dataset], models: List[nn.Module]):
        self.config = config
        self.trainer_name = type(self).__name__
        
        self.train_dataset = datasets[0]
        
        self.decom = models[0]
        self.decom = self.decom.to(self.config.device)
        
        self.network = models[1]
        self.network.apply(inplace_relu)
        self.network = self.network.to(self.config.device)
        
        self.adapter = models[2]
        if self.adapter is not None:
            self.adapter.apply(inplace_relu)
            self.adapter = self.adapter.to(self.config.device)
            
        self.offset = models[3]
        if self.offset is not None:
            self.offset.apply(inplace_relu)
            self.offset = self.offset.to(self.config.device)
            
        self.normalmap = models[4]
        if self.normalmap is not None:
            self.normalmap = self.normalmap.to(self.config.device)
            
        self.init_checkpoint()
        self.init_log()

    def init_log(self):
        log_path = os.path.join(
            self.config.root,
            self.config.save_root,
            self.start_time,
        )
        self.log_file = os.path.join(log_path, 'log.txt')
        if self.config.log_file:
            if not os.path.exists(log_path):
                os.system(f'mkdir -p {log_path}')
            os.system(f'mkdir -p {log_path}/code')
            os.system(f'cp -a {os.path.dirname(os.path.realpath(__file__))}/*.py {log_path}/code')
        self.log(self.config.to_lines(), output='file')
        self.config.print_to_screen()
    
    def log(self, line='', newline='', endline='\n', output='both'):
        '''
            log file helper function
        '''

        if not output in ['both', 'console', 'file']:
            raise ValueError('[Trainer: log] Unknown output direction.')

        if isinstance(line, list):
            if output in ['both', 'console']:
                for l in line:
                    print(newline, l, end=endline)
            if self.config.log_file and output in ['both', 'file']:
                with open(self.log_file, 'ab') as f:
                    np.savetxt(f, line, fmt='%s')
        elif isinstance(line, str):
            if output in ['both', 'console']:
                print(newline + line, end=endline)
            if self.config.log_file and output in ['both', 'file']:
                with open(self.log_file, 'ab') as f:
                    np.savetxt(f, [line], fmt='%s')
        else:
            if output in ['both', 'console']:
                print(f'{line}')
            if self.config.log_file and output in ['both', 'file']:
                with open(self.log_file, 'ab') as f:
                    np.savetxt(f, [line], fmt='%s')

    def show(self, obj, length=100):
        if isinstance(obj, nn.Module):
            for name, params in obj.named_parameters():
                if params.grad is not None:
                    line = f'>>  module {name}\n\tgrad: {params.grad.mean():.1e} +- ({params.grad.std():.1e})\t'\
                        f'param: {params.detach().cpu().numpy().mean():.1e} +- ({params.detach().cpu().numpy().std():.1e})'
                    self.log(line, output='console')
                else:
                    line = f'>>  module {name}\n\t\t\t\t\t'\
                        f'param: {params.detach().cpu().numpy().mean():.1e} +- ({params.detach().cpu().numpy().std():.1e})'
                    self.log(line, output='console')
        elif isinstance(obj, torch.Tensor):
            for i in range(min(obj.shape[0], length)):
                line = f'>>  tensor {i}\t{obj[i].detach().cpu().numpy().mean():.1e} +- ({obj[i].detach().cpu().numpy().std():.1e})'
                self.log(line, output='console')

    def set_lr(self, lr, index):
        self.optimizer.param_groups[index]['lr'] = lr
        self.log(f'lr {index} changed -> {lr}')

    def update_lr(self, scale, optimizer=None, eps=1e-8):
        if optimizer is None:
            optimizer = self.optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = (param_group['lr'] - eps) * scale + eps

    def draw_loss_error_curves(self, train_loss_curve, validate_error_curve, epoch):
        '''
        draw loss and error curves

        Args:
            train_loss_curve, N curves (list of np.array, [epoch+1, N])
            validate_error_curve, N curves (list of np.array, [epoch+1, N])

        '''
        train_loss_curve = np.array(train_loss_curve)
        validate_error_curve = np.array(validate_error_curve)

        x0 = np.arange(0, epoch)
        plt.clf()
        for i in range(train_loss_curve.shape[-1]):
            plt.plot(x0, [float(format(x, '.2g')) for x in train_loss_curve[:, i].reshape(-1).tolist()], label=f'train_loss_{i}')
        x1 = np.arange(self.config.validate_epoch-1, epoch, self.config.validate_epoch)
        if self.config.validate_epoch > 0 and validate_error_curve.shape[0] == x1.size:
            for i in range(validate_error_curve.shape[-1]):
                plt.plot(x1, [float(format(x, '.2g')) for x in validate_error_curve[:, i].reshape(-1).tolist()], label=f'validate_error_{i}')
        plt.xlabel('epoch')
        plt.ylabel('loss / error')
        title = self.start_time
        if len(self.start_time) > 50:
            break_point = (len(title.split('-')) + 1) // 2
            title = '-'.join(title.split('-')[:break_point]) + '\n-' + '-'.join(title.split('-')[-break_point:])
        plt.title(title)
        plt.grid(True)
        plt.legend()
        max_y = max(train_loss_curve.max(), validate_error_curve.max() if len(validate_error_curve) > 0 else 0)
        plt.yticks(np.arange(0, max_y, np.ceil(max_y) / 10))
        plt.savefig(os.path.join(self.config.root, self.config.save_root, self.start_time, 'loss.png'))
        plt.yscale('log')
        plt.savefig(os.path.join(self.config.root, self.config.save_root, self.start_time, 'loss_log.png'))

    
class Trainer(BaseTrainer):

    def __init__(self, config: RepConfig, datasets: List[torch.utils.data.Dataset], models: List[nn.Module]):
        super().__init__(config, datasets, models)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.smooth_l1loss = nn.SmoothL1Loss()

    def criterion(self, pred, gt):
        if self.config.compress_only:
            rgb_loss = self.mae_loss(pred, torch.clamp(gt, 1e-5, 1.0 - 1e-5))
        else:
            rgb_loss = self.mae_loss(pred, torch.clamp(gt, 0.0, 1.0))
        return rgb_loss, (rgb_loss.item(),)

    def init_checkpoint(self):
        '''
            normally: deal with start_time, init radius and optimizer, and finally deal with comments
            continue: load model and overwrite above
            compress only: after "continue", overwrite radius, optimizer and decom again
        '''
        
        self.start_time = self.config.start_time
        self.flag_param_changed = False
        self.radius = self.start_radius = self.config.start_radius
        self.optimizer = torch.optim.AdamW([
            {
                'params': self.decom.parameters(), 
                'lr': self.config.lr[0], 
                'weight_decay': 0.01,
            },
            {
                'params': self.network.parameters(), 
                'lr': self.config.lr[1],
                'weight_decay': 0.03,
            }
        ])
        if self.adapter is not None:
            self.adapter_optimizer = torch.optim.Adam([
                {
                    'params': self.adapter.parameters(),
                    'lr': self.config.lr[2],
                }
            ])
        if self.offset is not None:
            self.offset_optimizer = torch.optim.AdamW([
                {
                    'params': self.offset.offset_texture,
                    'lr': self.config.lr[3],
                    'weight_decay': 0,
                },
                {
                    'params': self.offset.offset_network.parameters(),
                    'lr': self.config.lr[4],
                    'weight_decay': 0.01,
                },
            ])
        if self.normalmap is not None:
            self.normalmap_optimizer = torch.optim.Adam([
                {
                    'params': self.normalmap.parameters(),
                    'lr': self.config.lr[5],
                }
            ])
            

        if self.config.continue_training:

            if self.config.checkpoint_path.startswith('/'):
                save_dict = torch.load(self.config.checkpoint_path)
            else:
                save_dict = torch.load(os.path.join(self.config.root, self.config.save_root, self.config.checkpoint_path))

            self.start_time = self.config.checkpoint_path.split('/')[-3].split('-')[-1] + '-' + self.config.checkpoint_path.split('/')[-1].replace('.pth', '') + '-' + self.start_time
            self.radius = self.start_radius = save_dict['radius']
            self.optimizer = save_dict['optimizer']
            optimizer_to(self.optimizer, self.config.device) ## sometimes the loaded opt params are in different device
            if self.adapter is not None and 'adapter_optimizer' in save_dict:
                self.adapter_optimizer = save_dict['adapter_optimizer']
                optimizer_to(self.adapter_optimizer, self.config.device)
            if self.offset is not None and 'offset_optimizer' in save_dict:
                self.offset_optimizer = save_dict['offset_optimizer']
            if self.normalmap is not None and 'normalmap_optimizer' in save_dict:
                self.normalmap_optimizer = save_dict['normalmap_optimizer']
                
            if '.state_dict' in self.config.checkpoint_path:
                self.network.decoder.load_state_dict(save_dict['decoder']())
                self.decom.load_state_dict(save_dict['decom']())
                if self.adapter is not None and 'adapter' in save_dict:
                    self.adapter.load_state_dict(save_dict['adapter']())
                if self.offset is not None and 'offset' in save_dict:
                    self.offset.load_state_dict(save_dict['offset']())
                if self.normalmap is not None and 'normalmap' in save_dict:
                    self.normalmap.load_state_dict(save_dict['normalmap']())
            else:
                self.network.decoder = save_dict['decoder']
                self.decom = save_dict['decom']
                if self.adapter is not None and 'adapter' in save_dict:
                    self.adapter = save_dict['adapter']
                if self.offset is not None and 'offset' in save_dict:
                    self.offset = save_dict['offset']
                if self.normalmap is not None and 'normalmap' in save_dict:
                    self.normalmap = save_dict['normalmap']
            self.network.to(self.config.device) ## sometimes the saved models are in different devices with the current work
            self.decom.to(self.config.device) 
            if self.adapter is not None:
                self.adapter.to(self.config.device) 
            if self.offset is not None:
                self.offset.to(self.config.device)
            if self.normalmap is not None:
                self.normalmap.to(self.config.device)
        
        if self.config.compress_only:
            
            self.radius = self.start_radius = self.config.start_radius
            self.decom = getattr(model, self.config.decom)(self.config, init='zeros')
            if self.adapter is not None:
                self.adapter = getattr(model, self.config.adapter)(self.config).to(self.config.device)
            if self.offset is not None:
                self.offset = getattr(model, self.config.offset)(self.config).to(self.config.device)
            if self.normalmap is not None:
                self.normalmap = getattr(model, self.config.normalmap)(self.config).to(self.config.device)
            self.optimizer = torch.optim.AdamW([
                {
                    'params': self.decom.parameters(), 
                    'lr': self.config.lr[0], 
                },
                {
                    'params': self.network.parameters(), 
                    'lr': self.config.lr[1], ## this should be 0
                }
            ])
            if self.adapter is not None:
                self.adapter_optimizer = torch.optim.Adam([
                    {
                        'params': self.adapter.parameters(),
                        'lr': 0 if self.config.change_parameters_epoch is not None else self.config.lr[2],
                    }
                ])
            if self.offset is not None:
                self.offset_optimizer = torch.optim.AdamW([
                    {
                        'params': self.offset.offset_texture,
                        'lr': self.config.lr[3],
                        'weight_decay': 0,
                    },
                    {
                        'params': self.offset.offset_network.parameters(),
                        'lr': self.config.lr[4],
                        'weight_decay': 0.01,
                    },
                ])
            if self.normalmap is not None:
                self.normalmap_optimizer = torch.optim.Adam([
                    {
                        'params': self.normalmap.parameters(),
                        'lr': self.config.lr[5],
                    }
                ])
                
                
            ## freeze network weights
            for pname, param in self.network.named_parameters():
                param.requires_grad = False
                
            ## re-activate the trainable layers
            if self.config.reacitvate_trainale_layers:
                for mname, module in self.network.named_modules(): 
                    if isinstance(module, trainable_layer): 
                        for pname, param in module.named_parameters():
                            param.requires_grad = True

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.lr_decay_param)
        if self.adapter is not None:
            self.adapter_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.adapter_optimizer, gamma=self.config.lr_decay_param)
        if self.offset is not None:
            self.offset_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.offset_optimizer, gamma=self.config.lr_decay_param)
        if self.normalmap is not None:
            self.normalmap_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.normalmap_optimizer, gamma=self.config.lr_decay_param)
            
        if self.config.comments != '':
            self.start_time = self.config.comments + '-' + self.start_time

    def save_model(self, network, decom, adapter, offset, normalmap, epoch, save_name, use_state_dict=False):
        path = os.path.join(self.config.root, self.config.save_root, save_name)
        save_dir = f'epoch-{epoch}'
        model_name = f'{save_dir}{".state_dict" if use_state_dict else ""}.pth'

        if not os.path.exists(os.path.join(path, save_dir)):
            os.system(f"mkdir -p {os.path.join(path, save_dir)}")

        save_dict = {
            'config': self.config,
            'optimizer': self.optimizer,
            'radius': self.radius,
            'epoch': epoch
        }

        if network is not None:
            save_dict['decoder'] = network.decoder.state_dict if use_state_dict else network.decoder
        if decom is not None:
            save_dict['decom'] = decom.state_dict if use_state_dict else decom
        if adapter is not None:
            save_dict['adapter'] = adapter.state_dict if use_state_dict else adapter
            save_dict['adapter_optimizer'] = self.adapter_optimizer
        if offset is not None:
            save_dict['offset'] = offset.state_dict if use_state_dict else offset            
            save_dict['offset_optimizer'] = self.offset_optimizer            
        if normalmap is not None:
            save_dict['normalmap'] = normalmap.state_dict if use_state_dict else normalmap
            save_dict['normalmap_optimizer'] = self.normalmap_optimizer
        
        torch.save(save_dict, os.path.join(path, save_dir, model_name))
        self.log(f'checkpoint {epoch} saved into {path}/{save_dir}/{model_name}')

    def get_radius(self, epoch):
        if self.config.radius_decay is None or self.config.radius_decay <= 0:
            self.radius = 0
        else:
            self.radius = int(np.floor((self.start_radius * (self.config.radius_decay ** (epoch-1)))))
        return self.radius

    def train(self):
        
        assert(isinstance(self.network, nn.Module))
        self.network.train()
        train_loss_curve = []

        if self.config.compress_only:
            self.log('compress only mode.', output='console')
        else:
            self.log('training mode.', output='console')
        self.log('decom & network lr -> ' + ', '.join([f'{param["lr"]:.1e}' for param in self.optimizer.param_groups]))
        if self.adapter is not None:
            self.log('adapter lr -> ' + ', '.join([f'{param["lr"]:.1e}' for param in self.adapter_optimizer.param_groups]))
        if self.offset is not None:
            self.log('offset lr -> ' + ', '.join([f'{param["lr"]:.1e}' for param in self.offset_optimizer.param_groups]))
        if self.normalmap is not None:
            self.log('normalmap lr -> ' + ', '.join([f'{param["lr"]:.1e}' for param in self.normalmap_optimizer.param_groups]))
        start_time = clock()

        epoch = -1
        while True:
            epoch += 1
            epoch_loss = []
            
            pbar = tqdm.tqdm(self.train_dataset.dataloader)
            pbar.set_description(f'ep {epoch+1}/{self.config.max_epochs}')
            avg_loss = []
            color = None
            output = None
            for batch_index, (material_index, wi, cos, h, d, u, v, color) in enumerate(pbar):
                
                wi, cos, h, d, u, v, color = map(lambda x: x.to(self.config.device), [wi, cos, h, d, u, v, color])
                color = torch.clamp(color, 0.0, 1.0)
                # color = torch.log1p(color)
                
                ## train this batch
                if self.normalmap is not None:
                    h = self.normalmap(material_index, h, u, v)
                if self.offset is not None:
                    u, v, _ = self.offset(material_index, u, v, wi, radius=self.get_radius(epoch+1))
                latent = self.decom(material_index, h, u, v, radius=self.get_radius(epoch+1)) ## size: [bs, latent_size]
                output = self.network(torch.cat([d, latent], dim=-1).reshape(-1, self.config.query_size + sum(self.config.latent_size[:2]))) ## size: [bs, 4+latent_size] -> [bs, 3]
                if self.adapter is not None:
                    output = self.adapter(output, material_index, u, v, radius=0)

                loss, division = self.criterion(output * cos.reshape(-1, 1), color.reshape(-1, 3))
                avg_loss.append(division)

                self.optimizer.zero_grad()
                if self.adapter is not None:
                    self.adapter_optimizer.zero_grad()
                if self.offset is not None:
                    self.offset_optimizer.zero_grad()
                if self.normalmap is not None:
                    self.normalmap_optimizer.zero_grad()
                    
                loss.backward(retain_graph=True)
                
                self.optimizer.step()
                if self.adapter is not None:
                    self.adapter_optimizer.step()
                if self.offset is not None:
                    self.offset_optimizer.step()
                if self.normalmap is not None:
                    self.normalmap_optimizer.step()

                if batch_index % max(len(self.train_dataset) // self.config.batch_size // 100, 1) == 0:
                    if self.config.log_file:
                        line = (
                            f'[{time_change(int(clock() - start_time))}] '
                            f'\tep {epoch+1}, {batch_index}/{self.train_dataset.__len__() // self.config.batch_size}, '
                            f'\tloss: {", ".join([f"{l:.1e}" for l in np.mean(avg_loss, 0)])}, '
                            f'\tnetwork grad: {np.mean([p.grad.mean().item() for p in self.network.parameters() if p.grad is not None]) if not self.config.compress_only else 0.0 :.1e} +- ' 
                            f'({np.std([p.grad.mean().item() for p in self.network.parameters() if p.grad is not None]) if not self.config.compress_only else 0.0 :.1e}), ' 
                            f'\tnetwork weights: {np.mean([p.mean().item() for p in self.network.parameters()]):.1e} +- '
                            f'({np.std([p.mean().item() for p in self.network.parameters()]):.1e}), '
                            f'\tdecom grad: {np.mean([p.grad.mean().item() for p in self.decom.parameters() if p.grad is not None]):.1e} +- '
                            f'({np.std([p.grad.mean().item() for p in self.decom.parameters() if p.grad is not None]):.1e}), '
                            f'\tdecom weights: {np.mean([p.mean().item() for p in self.decom.parameters()]):.1e} +- '
                            f'({np.std([p.mean().item() for p in self.decom.parameters()]):.1e}), '
                            f'\t{f"adapter weights: {np.mean([p.mean().item() for p in self.adapter.parameters()]):.1e} +- " if self.adapter is not None else ""}'
                            f'{f"({np.std([p.mean().item() for p in self.adapter.parameters()]):.1e}), " if self.adapter is not None else ""}'
                            f'\t{f"offset tex weights: {np.mean(self.offset.offset_texture.detach().cpu().numpy()):.1e} +- " if self.offset is not None else ""}'
                            f'{f"({np.std(self.offset.offset_texture.detach().cpu().numpy()):.1e}), " if self.offset is not None else ""}'
                            f'\t{f"offset net weights: {np.mean([p.mean().item() for p in self.offset.offset_network.parameters()]):.1e} +- " if self.offset is not None else ""}'
                            f'{f"({np.std([p.mean().item() for p in self.offset.offset_network.parameters()]):.1e}), " if self.offset is not None else ""}'
                            f'\t{f"normals: {np.mean([p.mean().item() for p in self.normalmap.parameters()]):.1e} +- " if self.normalmap is not None else ""}'
                            f'{f"({np.std([p.mean().item() for p in self.normalmap.parameters()]):.1e}), " if self.normalmap is not None else ""}'
                            f'\tradius: {self.radius if self.config.start_radius > 0 else "None"}'
                        )
                        self.log(line, newline='\r', endline='', output='file')
                    pbar.set_postfix(loss=f'{", ".join([f"{l:.1e}" for l in np.mean(avg_loss, 0)])}{f", radius={self.radius}" if self.config.start_radius != 0 else ""}')
                    epoch_loss.append(np.mean(avg_loss, 0))
                    avg_loss = []

                if os.path.exists('DEBUG.bat'):
                    print('DEBUG switch triggered.')
                    torch.cuda.empty_cache()
                    debug()
                    # del loss, output # prevent OOM error
                    
            ## end of an epoch
            pbar.close()
            if self.config.log_file:

                self.log(output='file')  ## every epoch starts a new line
                
                ## curves and output images
                train_loss_curve.append(np.mean(epoch_loss, 0))
                self.draw_loss_error_curves(train_loss_curve, [], epoch+1)
                if color is not None and output is not None:
                    exr.write(color.cpu().numpy().reshape(self.config.cache_file_shape[1], -1, 3)[:, :self.config.cache_file_shape[1]], 'color.exr')
                    exr.write(output.detach().cpu().numpy().reshape(self.config.cache_file_shape[1], -1, 3)[:, :self.config.cache_file_shape[1]], 'output.exr')
                if len(self.config.train_materials) == 1 and self.offset is not None:
                    self.offset.output_depth(wi=[0, 0.707, 0.707], out_path='depth.exr')
                if self.normalmap is not None:
                    self.normalmap.output_normalmap(out_path='normal.exr')

                ## validate and save model
                if ( 
                    epoch % self.config.validate_epoch == self.config.validate_epoch - 1 or 
                    epoch == self.config.max_epochs - 1 or
                    self.config.change_parameters_epoch is not None and epoch == self.config.change_parameters_epoch - 1
                ):
                    out_dir = os.path.join(self.config.root, self.config.save_root, self.start_time, f'epoch-{epoch+1}')
                    os.system('mkdir -p {}'.format(out_dir))
                    self.render_jobs(out_dir)
                    ## save model
                    if self.config.save_model: ## compress only doesn't need to save the model 
                        self.decom.vis_feature(out_dir, epoch+1)
                        if self.config.compress_only and self.adapter is not None:
                            self.adapter.vis_feature(out_dir, epoch+1)
                        if self.offset is not None:
                            self.offset.vis_feature(out_dir, epoch+1)
                        if self.normalmap is not None:
                            self.normalmap.vis_feature(out_dir, epoch+1)
                        self.save_model(self.network, self.decom, self.adapter, self.offset, self.normalmap, epoch+1, save_name=self.start_time)

            ## update lr
            if epoch % self.config.lr_decay_epoch == self.config.lr_decay_epoch - 1:
                self.lr_scheduler.step()
                if self.adapter is not None:
                    self.adapter_lr_scheduler.step()
                if self.offset is not None:
                    self.offset_lr_scheduler.step()
                if self.normalmap is not None:
                    self.normalmap_lr_scheduler.step()
            ## change params
            if self.config.compress_only and self.config.change_parameters_epoch is not None and epoch+1 == self.config.change_parameters_epoch:
                self.flag_param_changed = True
                self.log('='*(os.get_terminal_size().columns // 2))
                self.log(' CHANGE PARAMS ')
                self.log('='*(os.get_terminal_size().columns // 2))
                self.optimizer.param_groups[0]['lr'] = 0
                self.optimizer.param_groups[1]['lr'] = self.config.lr[1]
                if self.adapter is not None:
                    self.adapter_optimizer.param_groups[0]['lr'] = self.config.lr[2]
                if self.offset is not None:
                    self.offset_optimizer.param_groups[0]['lr'] = 0
                    self.offset_optimizer.param_groups[1]['lr'] = 0
                if self.normalmap is not None:
                    self.normalmap_optimizer.param_groups[0]['lr'] = 0
            ## output optim info
            self.log('decom & network lr -> ' + ', '.join([f'{param["lr"]:.1e}' for param in self.optimizer.param_groups]))
            if self.adapter is not None:
                self.log('adapter lr -> ' + ', '.join([f'{param["lr"]:.1e}' for param in self.adapter_optimizer.param_groups]))
            if self.offset is not None:
                self.log('offset lr -> ' + ', '.join([f'{param["lr"]:.1e}' for param in self.offset_optimizer.param_groups]))
            if self.normalmap is not None:
                self.log('normalmap lr -> ' + ', '.join([f'{param["lr"]:.1e}' for param in self.normalmap_optimizer.param_groups]))

            self.log(output='file')
            if epoch == self.config.max_epochs - 1:
                break
            
        ## end of training
        if self.config.log_file:
            os.system(f'mv {os.path.join(self.config.root, self.config.save_root, self.start_time)} {os.path.join(self.config.root, self.config.save_root, "#" + self.start_time)} ')
        print('all finished.')

    def render_jobs(self, out_dir):
        args = [self.network, self.decom, self.adapter, self.offset, self.normalmap, 0, self.config]
        with tqdm.tqdm(self.config.train_materials) as pbar:
            for i, m in enumerate(pbar):
                args[-2] = i
                pbar.set_description(f'rendering {m}:')
                # render(*args, './buffer_plane_point_persp.exr', os.path.join(out_dir, f'plane_train_{m}.exr'), self.config.device, output_brdf_value=False)
                render(*args, './buffer_plane_collocated.exr', os.path.join(out_dir, f'plane_train_col_{m}.exr'), self.config.device, output_brdf_value=False)
                # render(*args, './buffer_sphere_dir_orth.exr', os.path.join(out_dir, f'sphere_train_BRDFval_{m}.exr'), self.config.device, output_brdf_value=True)
                # render(*args, './buffer_plane_parallax.exr', os.path.join(out_dir, f'plane_train_parallax_{m}.exr'), self.config.device, output_brdf_value=False)
                # render(*args, './buffer_plane_collocated.exr', os.path.join(out_dir, f'plane_train_col_{m}_down.exr'), self.config.device, view=[0, 0.707, 0.707], output_brdf_value=False)
                # render(*args, './buffer_plane_collocated.exr', os.path.join(out_dir, f'plane_train_col_{m}_right.exr'), self.config.device, view=[0.707, 0, 0.707], output_brdf_value=False)
                # render(*args, './buffer_plane_collocated.exr', os.path.join(out_dir, f'plane_train_col_{m}_up.exr'), self.config.device, view=[0, -0.707, 0.707], output_brdf_value=False)
                # render(*args, './buffer_plane_collocated.exr', os.path.join(out_dir, f'plane_train_col_{m}_left.exr'), self.config.device, view=[-0.707, 0, 0.707], output_brdf_value=False)
