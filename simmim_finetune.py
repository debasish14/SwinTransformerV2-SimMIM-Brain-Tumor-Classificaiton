# --------------------------------------------------------
# SimMIM Fine-tuning for Brain Tumor Classification
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.cuda.amp as amp

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from brain_tumor_dataset import build_classification_loader
from swinv2_base_simmim_ft_config import get_config
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils_simmim import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, \
    reduce_tensor

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_option():
    parser = argparse.ArgumentParser('SimMIM fine-tuning script for Brain Tumor Classification', add_help=False)
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--epochs', type=int, default=50, help="number of training epochs")
    parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--enable-amp', action='store_true')
    parser.add_argument('--disable-amp', action='store_false', dest='enable_amp')
    parser.set_defaults(enable_amp=True)
    parser.add_argument('--output', default='output/simmim_finetune', type=str, metavar='PATH',
                        help='root of output folder, the full path is <o>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default='swinv2_base_ft', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--device', default='cuda', type=str, 
                        help='device to use for training / testing. cuda, mps or cpu')
    
    # Dataset settings
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--img-size', type=int, help='input image size')
    parser.add_argument('--num-workers', type=int, help='number of data loading workers')
    
    # Model settings
    parser.add_argument('--drop-rate', type=float, help='dropout rate')
    parser.add_argument('--drop-path-rate', type=float, help='drop path rate')
    parser.add_argument('--label-smoothing', type=float, help='label smoothing rate')
    
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
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
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
    if args.eval:
        config.EVAL_MODE = True
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.img_size:
        config.DATA.IMG_SIZE = args.img_size
    if args.num_workers:
        config.DATA.NUM_WORKERS = args.num_workers
    if args.drop_rate is not None:
        config.MODEL.DROP_RATE = args.drop_rate
    if args.drop_path_rate is not None:
        config.MODEL.DROP_PATH_RATE = args.drop_path_rate
    if args.label_smoothing is not None:
        config.MODEL.LABEL_SMOOTHING = args.label_smoothing
    if args.lr:
        config.TRAIN.BASE_LR = args.lr
    if args.min_lr:
        config.TRAIN.MIN_LR = args.min_lr
    if args.warmup_epochs:
        config.TRAIN.WARMUP_EPOCHS = args.warmup_epochs
    if args.weight_decay:
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
        
    # Set local rank
    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        config.LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))

    return config


def main():
    config = parse_option()
    
    # Make output directory
    os.makedirs(config.OUTPUT, exist_ok=True)
    
    # Set device
    device = torch.device(config.device)
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    elif config.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = torch.device("cpu")
    
    # Initialize distributed training if available
    if config.device == 'cuda' and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    distributed = False
    if config.device == 'cuda':
        try:
            torch.cuda.set_device(config.LOCAL_RANK)
            torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            torch.distributed.barrier()
            distributed = True
        except:
            print("No distributed training available, running in single GPU mode")
    
    # Set cudnn benchmark
    if config.device == 'cuda':
        cudnn.benchmark = True
    
    # Create logger
    from logger import create_logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank() if distributed else 0, name=f"{config.MODEL.NAME}")
    
    # Create data loaders
    logger.info("Creating dataset")
    train_dataset, val_dataset, data_loader_train, data_loader_val = build_classification_loader(
        root_dir=config.DATA.DATA_PATH,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS
    )
    
    # Update number of classes in config
    config.defrost()
    config.MODEL.NUM_CLASSES = len(train_dataset.classes)
    config.freeze()
    
    # Create model
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=False)
    model.to(device)
    logger.info(str(model))
    
    # Create optimizer
    optimizer = build_optimizer(config, model, simmim=True, is_pretrain=False)
    
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
        # PyTorch 2.0+ has built-in AMP support without explicit scaler
        scaler = None
    else:
        # For PyTorch 1.x, use GradScaler
        scaler = amp.GradScaler() if config.ENABLE_AMP else None
    
    # Create loss function
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize max accuracy
    max_accuracy = 0.0
    
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
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, criterion, logger)
        logger.info(f"Accuracy of the network on the {len(val_dataset)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return
    
    # Load pretrained model if specified
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, criterion, logger)
        logger.info(f"Accuracy of the network on the {len(val_dataset)} test images: {acc1:.1f}%")
    
    # Start training
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, scaler, logger)
        
        if (dist.get_rank() == 0 if distributed else True) and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, scaler, logger)
        
        acc1, acc5, loss = validate(config, data_loader_val, model, criterion, logger)
        logger.info(f"Accuracy of the network on the {len(val_dataset)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, scaler, logger):
    model.train()
    optimizer.zero_grad()
    
    logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_scale_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    device = next(model.parameters()).device
    
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with amp.autocast(enabled=config.ENABLE_AMP):
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        acc1, acc5 = accuracy(outputs, targets, topk=(1, min(5, config.MODEL.NUM_CLASSES)))
        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            
            if PYTORCH_MAJOR_VERSION >= 2 or not config.ENABLE_AMP:
                # PyTorch 2.0+ or no AMP
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
                # PyTorch 1.x with AMP
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
                # PyTorch 2.0+ or no AMP
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                # PyTorch 1.x with AMP
                scaler.scale(loss).backward()
                if config.TRAIN.CLIP_GRAD:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step_update(epoch * num_steps + idx)
        
        torch.cuda.synchronize()
        
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        if scaler is not None:
            loss_scale_meter.update(scaler.get_scale())
        else:
            loss_scale_meter.update(1.0)  # No scaling in PyTorch 2.0+ or when AMP is disabled
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {loss_scale_meter.val:.4f} ({loss_scale_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def validate(config, data_loader, model, criterion, logger):
    model.eval()
    
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    end = time.time()
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for idx, (images, target) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # compute output
            with amp.autocast(enabled=config.ENABLE_AMP):
                output = model(images)
                loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, config.MODEL.NUM_CLASSES)))
            
            if dist.is_initialized():  # For distributed training
                loss = reduce_tensor(loss)
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)
            
            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
    
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    main()
