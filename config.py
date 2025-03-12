from datetime import datetime
import os

from utils import pr

## to avoid cuda fork() error
# from multiprocessing import set_start_method
# set_start_method('spawn')

class BaseConfig():

    def print_to_screen(self, offset=0):
        lines = []
        term_width = os.get_terminal_size().columns - offset
        left_width = min(max([len(k) for k in vars(self).keys()]) + 2, term_width // 3)
        right_width = term_width - left_width
        lines.append('='*term_width)
        lines.append(f'[{self.__class__.__name__}]')
        lines.append('-'*term_width)
        if hasattr(self, "comments"):
            if len(self.comments) > right_width:
                comments = self.comments[:right_width // 2] + '...' + self.comments[-right_width + 4:]
            else:
                comments = self.comments
            lines.append(f'{"comments":<{left_width}} {comments}')
        for k,v in vars(self).items():
            if k in ['data_root', 'network', 'trainer', 'save_root', 'decom_params']:
                lines.append('-'*term_width)
            if k not in ['comments', 'other_info']:
                if len(k) > left_width:
                    k = k[:left_width // 2 - 4] + '...' + k[-left_width // 2:]
                if len(str(v)) > right_width:
                    v = str(v)[:right_width // 2] + '...' + str(v)[-right_width // 2 + 4:]
                lines.append(f'{k:<{left_width}} {v}')
        lines.append('='*term_width)
        pr('\n'.join(lines))
        return lines
    
    def to_lines(self):
        lines = []
        lines.append('='*80)
        lines.append(f'[{self.__class__.__name__}]')
        lines.append('-'*80)
        if hasattr(self, "comments"):
            lines.append(f'{"comments":<30} {self.comments}')
        for k,v in vars(self).items():
            if k in ['data_root', 'network', 'trainer', 'save_root', 'decoder_params']:
                lines.append('')
            if k not in ['comments', 'other_info']:
                lines.append(f'{k:<30} {v}')
        lines.append('='*80)
        return lines
        
    def __repr__(self) -> str:
        string = ''
        for line in self.to_lines():
            string += line + '\n'
        return string
        
class RepConfig(BaseConfig):

    def __init__(self, device, global_random_seed):

        self.other_info = 'release_test'
        self.start_time = datetime.now().strftime(r'%m%d_%H%M%S')
        self.random_seed = global_random_seed
        self.device = device
        self.root = '/test/repositories/mitsuba-pytorch-biplane/'

        ## data config
        self.data_root = {
            'default':  'data/',
        }
        self.train_dataset = 'TrainDataset_allrandom'
        self.btf_size = [400, 400] 
        self.cache_file_shape = [400, 400, 400, 9]
        self.train_materials = [
            'synthetic_rock2',
            # 'captured_009',
        ]

        ## model config
        self.network = 'Decoder'
        self.decom = 'DualBiPlane'
        self.adapter = 'Adapter'
        self.offset = None#'Offset'
        self.normalmap = None#'NormalMap'
        self.reacitvate_trainale_layers = False
        self.query_size = 2
        self.greyscale = False
        self.latent_size = [6, 6, 6]
        self.decom_H_reso = 20
        
        ## training config
        self.trainer = 'Trainer'
        self.batch_size = 8
        self.random_drop_queries = 2
        self.num_workers = min(self.batch_size * 2, os.cpu_count())
        self.lr = [1e-3, 3e-4, 0, 0, 0, 0] ## decom network adapter offset_tex offset_network normalmap
        self.lr_decay_param = 0.9
        self.lr_decay_epoch = 1
        self.start_radius = 8
        self.radius_decay = 0.8
        self.max_epochs = 40
        self.validate_epoch = 5
        self.change_parameters_epoch = None
        
        ## checkpoint config
        self.save_root = 'torch/saved_model'
        self.log_file = self.save_model = True
        self.compress_only = True ## if only optimize the representations. If True, the network will be frozen.
        self.continue_training = True ## if load any checkpoints
        if self.compress_only: 
            ## Optimization parameters for latent representations
            self.batch_size = 8
            self.random_drop_queries = 2
            self.num_workers = min(self.batch_size * 2, os.cpu_count())
            self.start_radius = 20
            self.radius_decay = 0.75
            self.use_hxy_comb = False ## use h_plane combination for catpured photos. For synthetic data, set to False
            self.lr = [1e-2, 0, 1e-2, 1e-2, 3e-3, 1e-2]
            self.lr_decay_param = 0.85
            self.max_epochs = 20
            self.lr_decay_epoch = 1
            self.change_parameters_epoch = 15
            self.validate_epoch = -1
        self.checkpoint_path = 'saved_model/epoch-35.pth'

        import model
        self.dummy_decom = getattr(model, self.decom)(self)
        self.decom_params = self.dummy_decom.get_param_count()
        self.dummy_model = getattr(model, self.network)(self)
        self.decoder_params = self.dummy_model.get_param_count()
        del self.dummy_decom, self.dummy_model 
        if self.adapter:
            self.dummy_adapter = getattr(model, self.adapter)(self)
            self.adapter_params = self.dummy_adapter.get_param_count()
            del self.dummy_adapter
        if self.offset:
            self.dummy_offset = getattr(model, self.offset)(self)
            self.offset_params = self.dummy_offset.get_param_count()
            del self.dummy_offset
        if self.normalmap:
            self.dummy_normalmap = getattr(model, self.normalmap)(self)
            self.normalmap_params = self.dummy_normalmap.get_param_count()
            del self.dummy_normalmap
        
        self.gen_comment()

    def gen_comment(self):

        ## other info
        self.comments = self.other_info
        if self.compress_only:
            self.comments = 'compress_only' + ('-' if self.other_info else '') + self.other_info
            ## data name
            if len(self.train_materials) < 3:
                self.comments += f'-{"_".join(self.train_materials)}'
            return

        ## model name / structure
        self.comments += f'-{self.network}-{self.decom}'
        
        ## latent structure
        self.comments += f'-H{self.decom_H_reso}^2_L{self.latent_size[0]}+{self.latent_size[1]}'
        
        ## data name
        if len(self.train_materials) < 3:
            self.comments += f'-{"_".join(self.train_materials)}'

        ## training settings
        self.comments += f'-bs{self.batch_size}'
        
        ## dataset
        self.comments += f'-{self.cache_file_shape[0]}x{len(self.train_materials)}BTF' + ('s' if len(self.train_materials) > 1 else '')
        
if __name__ == '__main__':
            
    config = RepConfig('cuda:0', 0)
    config.print_to_screen()
    