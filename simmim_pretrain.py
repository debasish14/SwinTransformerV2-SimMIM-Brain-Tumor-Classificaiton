# --------------------------------------------------------
# SimMIM Pre-training for Brain Tumor Dataset
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.cuda.amp
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.cuda.amp
from torch.utils.tensorboard import SummaryWriter

from timm.utils import AverageMeter

from brain_tumor_dataset import build_simmim_loader
from swinv2_base_simmim_pt_config import get_config
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils_simmim import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script for Brain Tumor Dataset', add_help=False)
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--epochs', type=int, default=100, help="number of training epochs")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--enable-amp', action='store_true')
    parser.add_argument('--disable-amp', action='store_false', dest='enable_amp')
    parser.set_defaults(enable_amp=True)
    parser.add_argument('--output', default='output/simmim_pretrain', type=str, metavar='PATH',
                        help='root of output folder, the full path is <o>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default='swinv2_base_pt', help='tag of experiment')
    parser.add_argument('--device', default='cuda', type=str, 
                        help='device to use for training / testing. cuda, mps or cpu')

    # Dataset settings
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--img-size', type=int, help='input image size')
    parser.add_argument('--num-workers', type=int, help='number of data loading workers')
    
    # SimMIM specific settings
    parser.add_argument('--mask-patch-size', type=int, help='mask patch size for SimMIM')
    parser.add_argument('--mask-ratio', type=float, help='masking ratio for SimMIM')
    parser.add_argument('--norm-target-enable', action='store_true', help='enable norm target')
    parser.add_argument('--norm-target-disable', action='store_false', dest='norm_target_enable')
    parser.add_argument('--norm-target-patch-size', type=int, help='norm target patch size')
    
    # Model settings
    parser.add_argument('--drop-rate', type=float, help='dropout rate')
    parser.add_argument('--drop-path-rate', type=float, help='drop path rate')
    
    # Training settings
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--min-lr', type=float, help='minimum learning rate')
    parser.add_argument('--warmup-epochs', type=int, help='epochs to warmup LR')
    parser.add_argument('--weight-decay', type=float, help='weight decay')

    # distributed training
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=False, default=0, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config()
    
    # Update config from arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.TRAIN.EPOCHS = args.epochs
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = args.use_checkpoint
    if args.enable_amp is not None:
        config.ENABLE_AMP = args.enable_amp
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
        
    # Add device parameter to config
    config.DEVICE = args.device
        
    # Set local rank
    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        config.LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))

    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.img_size:
        config.DATA.IMG_SIZE = args.img_size
    if args.num_workers:
        config.DATA.NUM_WORKERS = args.num_workers
    if args.mask_patch_size:
        config.DATA.MASK_PATCH_SIZE = args.mask_patch_size
    if args.mask_ratio is not None:
        config.DATA.MASK_RATIO = args.mask_ratio
    if args.norm_target_enable is not None:
        config.NORM_TARGET.ENABLE = args.norm_target_enable
    if args.norm_target_patch_size:
        config.NORM_TARGET.PATCH_SIZE = args.norm_target_patch_size
    if args.drop_rate is not None:
        config.MODEL.DROP_RATE = args.drop_rate
    if args.drop_path_rate is not None:
        config.MODEL.DROP_PATH_RATE = args.drop_path_rate
    if args.lr:
        config.TRAIN.BASE_LR = args.lr
    if args.min_lr:
        config.TRAIN.MIN_LR = args.min_lr
    if args.warmup_epochs:
        config.TRAIN.WARMUP_EPOCHS = args.warmup_epochs
    if args.weight_decay:
        config.TRAIN.WEIGHT_DECAY = args.weight_decay

    return config


def main():
    config = parse_option()
    
    # Make output directory
    os.makedirs(config.OUTPUT, exist_ok=True)
    
    # Set device
    device = torch.device(config.DEVICE)
    if config.DEVICE == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    elif config.DEVICE == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = torch.device("cpu")
    
    # Initialize distributed training if available
    if config.DEVICE == 'cuda' and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    distributed = False
    if config.DEVICE == 'cuda':
        try:
            torch.cuda.set_device(config.LOCAL_RANK)
            torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            torch.distributed.barrier()
            distributed = True
        except:
            print("No distributed training available, running in single GPU mode")
    
    # Set cudnn benchmark
    if config.DEVICE == 'cuda':
        cudnn.benchmark = True
    
    # Create logger
    from logger import create_logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank() if distributed else 0, name=f"{config.MODEL.NAME}")
    
    # Initialize tensorboard writer
    if dist.get_rank() == 0:
        writer = SummaryWriter(os.path.join(config.OUTPUT, 'tensorboard'))
    else:
        writer = None
    
    # Create data loader
    logger.info("Creating dataset")
    data_loader_train = build_simmim_loader(
        root_dir=config.DATA.DATA_PATH,
        img_size=config.DATA.IMG_SIZE,
        mask_patch_size=config.DATA.MASK_PATCH_SIZE,
        model_patch_size=config.MODEL.SWINV2.PATCH_SIZE,
        mask_ratio=config.DATA.MASK_RATIO,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS
    )
    
    # Create model
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.to(device)
    logger.info(str(model))
    
    # Create optimizer
    optimizer = build_optimizer(config, model, simmim=True, is_pretrain=True)
    
    # Wrap model with DistributedDataParallel if distributed training is enabled
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module if distributed else model
    
    # Log parameters and FLOPs
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    
    # Create learning rate scheduler
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    # Create gradient scaler for AMP
    if PYTORCH_MAJOR_VERSION >= 2:
        scaler = None
    else:
        scaler = torch.amp.GradScaler() if config.ENABLE_AMP else None
    
    # Auto resume if enabled
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    
    # Load checkpoint if resume is specified
    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger)
    
    # Start training
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, scaler, logger, writer)
        
        if (dist.get_rank() == 0 if distributed else True) and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, scaler, logger)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, scaler, logger, writer=None):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_scale_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    device = next(model.parameters()).device

    # Log sample images every 50 epochs
    log_images = epoch % 50 == 0 or epoch == 0
    
    for idx, (img, mask, _) in enumerate(data_loader):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        # Use device-agnostic autocast instead of CUDA-specific one
        with torch.amp.autocast(device_type=device.type, enabled=config.ENABLE_AMP):
            loss, pred = model(img, mask, return_pred=True)
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
            if PYTORCH_MAJOR_VERSION >= 2 or not config.ENABLE_AMP:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                scaler.scale(loss).backward()
                if config.TRAIN.CLIP_GRAD:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                    lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            
            if PYTORCH_MAJOR_VERSION >= 2 or not config.ENABLE_AMP:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                scaler.scale(loss).backward()
                if config.TRAIN.CLIP_GRAD:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step_update(epoch * num_steps + idx)
    
    # Only synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    loss_meter.update(loss.item(), img.size(0))
    norm_meter.update(grad_norm)
    if scaler is not None:
        loss_scale_meter.update(scaler.get_scale())
    else:
        loss_scale_meter.update(1.0)  
    batch_time.update(time.time() - end)
    end = time.time()
        
    if idx % config.PRINT_FREQ == 0:
        lr = optimizer.param_groups[0]['lr']
        memory_used = 0
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)
        logger.info(
            f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
            f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
            f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
            f'loss_scale {loss_scale_meter.val:.4f} ({loss_scale_meter.avg:.4f})\t'
            f'mem {memory_used:.0f}MB')
        
        # Log metrics to tensorboard
        if writer is not None and dist.get_rank() == 0:
            global_step = epoch * num_steps + idx
            writer.add_scalar('train/loss', loss_meter.val, global_step)
            writer.add_scalar('train/grad_norm', norm_meter.val, global_step)
            writer.add_scalar('train/loss_scale', loss_scale_meter.val, global_step)
            writer.add_scalar('train/lr', lr, global_step)
            
            # Log sample images and their reconstructions
            if log_images and idx == 0:
                # Get a sample batch
                sample_img = img[:4].detach().cpu()  # Take first 4 images
                sample_pred = pred[:4].detach().cpu()  # Take first 4 predictions
                sample_mask = mask[:4].detach().cpu()  # Take first 4 masks
                
                # Create grid of original images
                writer.add_images('train/original_images', sample_img, epoch)
                # Create grid of reconstructed images
                writer.add_images('train/reconstructed_images', sample_pred, epoch)
                # Create grid of masks
                writer.add_images('train/masks', sample_mask.unsqueeze(1).float(), epoch)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    
    # Log epoch metrics
    if writer is not None and dist.get_rank() == 0:
        writer.add_scalar('train/epoch_loss', loss_meter.avg, epoch)
        writer.add_scalar('train/epoch_grad_norm', norm_meter.avg, epoch)
        writer.add_scalar('train/epoch_loss_scale', loss_scale_meter.avg, epoch)


if __name__ == '__main__':
    main()
