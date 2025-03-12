import sys

import model
import trainer
import dataset
from utils import set_global_random_seed, detect_device
from config import RepConfig

def representation_main(device, global_random_seed, argv=None):
    
    config = RepConfig(device, global_random_seed)

    test_run = False
    if argv is not None and len(argv) > 1 and argv[1] == 'test_run':
        config.cache_file_shape[0] = config.batch_size * 4
        config.train_materials = config.train_materials[0:1]

        config.validate_epoch = 1
        config.change_parameters_epoch = 1

        config.log_file = config.save_model = True
        config.comments = 'TEST_RUN-' + config.comments
        test_run = True
    
    ## training
    decom = getattr(model, config.decom)(config)
    network = getattr(model, config.network)(config)
    adapter = getattr(model, config.adapter)(config) if config.adapter is not None else None
    offset = getattr(model, config.offset)(config) if config.offset is not None else None
    normalmap = getattr(model, config.normalmap)(config) if config.normalmap is not None else None
    train_dataset = getattr(dataset, config.train_dataset)(config, test_run=test_run)
    getattr(trainer, config.trainer)(
        config, [train_dataset], [decom, network, adapter, offset, normalmap]
    ).train()

if __name__ == '__main__':
        
    if sys.argv[-1] == 'debug':
        import debugpy
        debugpy.connect(('172.27.10.100', 5678))
        
    seed = set_global_random_seed(seed=None)
    device = detect_device()

    representation_main(device, seed, sys.argv)