# --------------------------------------------------------
# SimMIM pre-training configuration for Swin Transformer V2 on Brain Tumor Dataset
# --------------------------------------------------------

from config import get_config as get_base_config
from yacs.config import CfgNode as CN

def get_config():
    config = get_base_config()
    
    # Model settings
    config.MODEL.TYPE = 'swinv2'
    config.MODEL.NAME = 'swinv2_base_patch4_window7_224'
    config.MODEL.SWINV2 = config.MODEL.SWIN
    config.MODEL.SWINV2.PATCH_SIZE = 4
    config.MODEL.SWINV2.IN_CHANS = 3
    config.MODEL.SWINV2.EMBED_DIM = 128
    config.MODEL.SWINV2.DEPTHS = [2, 2, 18, 2]
    config.MODEL.SWINV2.NUM_HEADS = [4, 8, 16, 32]
    config.MODEL.SWINV2.WINDOW_SIZE = 7
    config.MODEL.SWINV2.MLP_RATIO = 4.
    config.MODEL.SWINV2.QKV_BIAS = True
    config.MODEL.SWINV2.APE = False
    config.MODEL.SWINV2.PATCH_NORM = True
    config.MODEL.DROP_RATE = 0.0
    config.MODEL.DROP_PATH_RATE = 0.1
    
    # Data settings
    config.DATA.DATASET = 'brain_tumor'
    config.DATA.DATA_PATH = '/Users/debasishborah/Downloads/brain-tumor-dataset'
    config.DATA.IMG_SIZE = 224
    config.DATA.MASK_PATCH_SIZE = 32
    config.DATA.MASK_RATIO = 0.6
    config.DATA.BATCH_SIZE = 32
    config.DATA.NUM_WORKERS = 8
    
    # Training settings
    config.TRAIN.START_EPOCH = 0
    config.TRAIN.EPOCHS = 100
    config.TRAIN.WARMUP_EPOCHS = 10
    config.TRAIN.WEIGHT_DECAY = 0.05
    config.TRAIN.BASE_LR = 1e-4
    config.TRAIN.WARMUP_LR = 1e-7
    config.TRAIN.MIN_LR = 1e-6
    config.TRAIN.CLIP_GRAD = 5.0
    config.TRAIN.AUTO_RESUME = True
    config.TRAIN.ACCUMULATION_STEPS = 1
    config.TRAIN.USE_CHECKPOINT = False
    
    # Augmentation settings
    config.AUG.COLOR_JITTER = 0.4
    
    # Norm target settings
    config.NORM_TARGET = CN()
    config.NORM_TARGET.ENABLE = True
    config.NORM_TARGET.PATCH_SIZE = 47
    
    # Output settings
    config.OUTPUT = 'output/simmim_pretrain'
    config.TAG = 'swinv2_base_pt'
    config.SAVE_FREQ = 10
    config.PRINT_FREQ = 100
    
    # Distributed training settings
    config.DIST_BACKEND = 'nccl'
    config.DIST_EVAL = True
    
    # Enable AMP
    config.ENABLE_AMP = True
    
    return config
