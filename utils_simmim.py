# --------------------------------------------------------
# SimMIM Utilities
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import datetime
import numpy as np
from collections import defaultdict, deque

def load_checkpoint(config, model, optimizer, lr_scheduler, scaler, logger):
    logger.info(f"==============> Resuming from {config.MODEL.RESUME}....................")
    if os.path.isfile(config.MODEL.RESUME):
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        max_accuracy = 0.0
        if 'model' in checkpoint:
            model_state_dict = checkpoint['model']
            if 'module' in list(model_state_dict.keys())[0]:
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"Checkpoint loaded with msg: {msg}")
        else:
            msg = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Checkpoint loaded with msg: {msg}")

        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Optimizer loaded")
        
        if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("LR Scheduler loaded")
        
        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
            logger.info("Scaler loaded")
        
        if 'epoch' in checkpoint:
            config.defrost()
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
            config.freeze()
            logger.info(f"Resume from epoch {checkpoint['epoch']}")
        
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
            logger.info(f"Max accuracy: {max_accuracy:.2f}%")
        
        return max_accuracy
    else:
        logger.warning(f"No checkpoint found at {config.MODEL.RESUME}")
        return 0.0

def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning...")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
        if 'module' in list(model_state_dict.keys())[0]:
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    else:
        model_state_dict = checkpoint
    
    # Delete head weights if num_classes don't match
    if hasattr(model, 'head') and hasattr(model.head, 'weight') and 'head.weight' in model_state_dict and model.head.weight.shape[0] != model_state_dict['head.weight'].shape[0]:
        logger.warning(f"Removing head weights from pretrained model (different number of classes)")
        del model_state_dict['head.weight']
        del model_state_dict['head.bias']
    
    msg = model.load_state_dict(model_state_dict, strict=False)
    logger.info(f"Missing keys: {msg.missing_keys}")
    logger.info(f"Unexpected keys: {msg.unexpected_keys}")
    
    return model

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, scaler, logger):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'max_accuracy': max_accuracy,
        'epoch': epoch,
    }
    if scaler is not None:
        save_state['scaler'] = scaler.state_dict()
    
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints found in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
