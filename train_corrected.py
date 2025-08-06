import torch
torch.cuda.empty_cache()
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import glob
import os
from tqdm import tqdm
from def_model_fixed import Transformer
import csv
import numpy as np
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数 - 防止过拟合的关键技术"""
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        pred: [N, C] 模型预测logits
        target: [N] 真实标签
        """
        pred = pred.log_softmax(dim=-1)
        
        # 创建平滑标签
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))  # 排除真实标签
        
        # 忽略padding token
        mask = (target != self.ignore_index)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * pred, dim=1)
        return torch.sum(loss * mask.float()) / mask.sum().clamp(min=1)

class OptimizedChunkDataset(Dataset):
    def __init__(self, chunk_dir):
        self.chunk_dir = chunk_dir
        self.enc_files = sorted(glob.glob(f"{chunk_dir}/enc_inputs_*.pt"))
        self.dec_in_files = sorted(glob.glob(f"{chunk_dir}/dec_inputs_*.pt"))
        self.dec_out_files = sorted(glob.glob(f"{chunk_dir}/dec_outputs_*.pt"))
        
        if not self.enc_files:
            raise ValueError(f"在 {chunk_dir} 中没有找到数据文件")
        
        self.chunk_sizes = []
        for f in self.enc_files:
            try:
                data = torch.load(f)
                self.chunk_sizes.append(data.shape[0])
            except Exception as e:
                print(f"警告：无法加载文件 {f}: {e}")
                self.chunk_sizes.append(0)
        
        self.total_samples = sum(self.chunk_sizes)
        self.cumulative_sizes = np.cumsum([0] + self.chunk_sizes)
        
        # 缓存当前加载的块
        self.current_chunk_idx = -1
        self.current_enc_data = None
        self.current_dec_in_data = None
        self.current_dec_out_data = None
        
        if dist.get_rank() == 0:
            print(f"数据集初始化完成：{len(self.enc_files)} 个块，总计 {self.total_samples} 个样本")
        
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        chunk_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[chunk_idx]
        
        if chunk_idx != self.current_chunk_idx:
            try:
                self.current_enc_data = torch.load(self.enc_files[chunk_idx])
                self.current_dec_in_data = torch.load(self.dec_in_files[chunk_idx])
                self.current_dec_out_data = torch.load(self.dec_out_files[chunk_idx])
                self.current_chunk_idx = chunk_idx
            except Exception as e:
                print(f"错误：无法加载块 {chunk_idx}: {e}")
                return torch.zeros(256, dtype=torch.long), torch.zeros(256, dtype=torch.long), torch.zeros(256, dtype=torch.long)
        
        return (self.current_enc_data[local_idx], 
                self.current_dec_in_data[local_idx], 
                self.current_dec_out_data[local_idx])

def get_lr_with_warmup(step_num, d_model=512, warmup_steps=4000):
    """
    原始论文的学习率调度 - 移除最大学习率限制
    """
    if step_num == 0:
        return 0.0
    
    # 论文原始公式
    arg1 = step_num ** (-0.5)
    arg2 = step_num * (warmup_steps ** (-1.5))
    
    return (d_model ** (-0.5)) * min(arg1, arg2)

def save_checkpoint(model, optimizer, epoch, global_step, loss, checkpoint_dir='checkpoints_regularized2'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{global_step}.pt')
    
    # 如果是DDP模型，保存时取出原始模型
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理DDP模型加载
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['global_step'], checkpoint['loss']

def find_latest_checkpoint(checkpoint_dir='checkpoints_regularized2'):
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: (
        int(x.split('_')[2]),
        int(x.split('_')[4].split('.')[0])
    ))
    return os.path.join(checkpoint_dir, checkpoints[-1])

def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def main(rank, world_size):
    # 初始化分布式训练
    setup(rank, world_size)
    
    # 设备设置
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"使用 {world_size} 个GPU进行训练")
    
    # 初始化模型
    model = Transformer().to(device)
    
    # 使用DDP包装模型
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if rank == 0:
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 获取词汇表大小
    from def_model_fixed import tgt_vocab_size
    if rank == 0:
        print(f"目标词汇表大小: {tgt_vocab_size}")
    
    # 使用标签平滑损失函数
    criterion = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size, 
        smoothing=0.1,
        ignore_index=0
    )
    
    # 优化器 - 添加权重衰减正则化
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.0,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-4
    )
    
    # 训练参数
    d_model = 512
    warmup_steps = 4000
    max_grad_norm = 1.0
    adaptive_grad_norm = True
    grad_norm_threshold = 5.0
    
    # 训练参数
    num_epochs = 5
    print_every = 100
    save_every = 5000
    
    if rank == 0:
        print(f"正则化设置:")
        print(f"  - 标签平滑: 0.1")
        print(f"  - 权重衰减: 1e-4") 
        print(f"  - 梯度裁剪: {max_grad_norm}")
        print(f"  - 自适应梯度调整: {'启用' if adaptive_grad_norm else '禁用'}")
        print(f"  - 梯度范数阈值: {grad_norm_threshold}")
        print(f"  - 使用原始论文学习率调度")
    
    # 恢复训练
    start_epoch = 0
    global_step = 0
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint and rank == 0:
        print(f"从检查点恢复: {latest_checkpoint}")
        try:
            start_epoch, global_step, _ = load_checkpoint(latest_checkpoint, model, optimizer, device)
            print(f"已恢复训练进度: 从第 {start_epoch} epoch, 第 {global_step} 步继续")
        except Exception as e:
            print(f"恢复检查点失败: {e}")
            start_epoch = 0
            global_step = 0
    
    # 确保所有进程同步
    dist.barrier()
    
    # 数据加载
    batch_size = 32  # 每个GPU的batch大小
    try:
        dataset = OptimizedChunkDataset('data_chunks_fixed')
    except ValueError as e:
        if rank == 0:
            print(f"数据加载失败: {e}")
            print("请先运行 train_data_fixed.py 生成数据")
        exit(1)
    
    # 使用DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    steps_per_epoch = len(train_loader)
    
    # 日志设置 (只在rank 0上记录)
    if rank == 0:
        os.makedirs('logs', exist_ok=True)
        loss_csv_path = 'logs/training_loss_regularized2.csv'
        
         # 只有当是首次训练或文件不存在时才写入表头
        if not os.path.exists(loss_csv_path) or global_step == 0:
            with open(loss_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'step', 'batch_loss', 'avg_loss', 'lr', 'grad_norm', 'status'])

        print(f"开始正则化训练 - 总样本数: {len(dataset)}, 每epoch步数: {steps_per_epoch}")
    steps_per_epoch = len(train_loader)
    # 计算恢复时应该从哪个epoch和step开始
    resume_epoch = global_step // steps_per_epoch
    resume_step_in_epoch = global_step % steps_per_epoch

    if rank == 0 and global_step > 0:
        print(f"恢复训练：将从 epoch {resume_epoch + 1}, batch {resume_step_in_epoch} 开始")

    
    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        sampler.set_epoch(epoch)  # 设置epoch保证shuffle正确
        
        epoch_loss = 0
        valid_batches = 0
        
        # 只在主进程显示进度条
        if rank == 0:
            initial_step = resume_step_in_epoch if epoch == resume_epoch else 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",initial=0, total=len(train_loader))
        else:
            progress_bar = train_loader
        
        for batch_idx, (enc_batch, dec_batch, tgt_batch) in enumerate(progress_bar):
            if epoch == resume_epoch and batch_idx < resume_step_in_epoch:
                if rank == 0:
                    progress_bar.update(1)
                continue
            try:
                # 基本数据验证
                if enc_batch.shape[0] != dec_batch.shape[0] or dec_batch.shape[0] != tgt_batch.shape[0]:
                    continue
                

                # 移动到设备
                enc_batch = enc_batch.to(device, non_blocking=True)
                dec_batch = dec_batch.to(device, non_blocking=True)
                tgt_batch = tgt_batch.to(device, non_blocking=True)

                # 前向传播
                outputs, *_ = model(enc_batch, dec_batch)
                
                # 计算标签平滑损失
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = tgt_batch.view(-1)
                loss = criterion(outputs_flat, targets_flat)
                
                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        print(f"警告: batch {batch_idx} 损失异常，跳过")
                    continue
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 计算梯度范数
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # 自适应梯度调整
                if adaptive_grad_norm and grad_norm > grad_norm_threshold:
                    if rank == 0 and global_step % print_every == 0:
                        print(f"⚠️  梯度范数过大 ({grad_norm:.2f}), 启用自适应调整")
                    # 如果梯度太大，使用更保守的更新
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad *= min(1.0, grad_norm_threshold / grad_norm)
                
                optimizer.step()
                
                # 学习率调度
                global_step += 1
                current_lr = get_lr_with_warmup(
                    global_step, 
                    d_model=d_model, 
                    warmup_steps=warmup_steps
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # 同步所有进程的损失
                reduced_loss = loss.detach().clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss = reduced_loss / world_size
                
                epoch_loss += reduced_loss.item()
                valid_batches += 1
                
                # 打印和记录 (只在rank 0上执行)
                if rank == 0 and global_step % print_every == 0:
                    avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
                    grad_status = "ADAPTIVE" if adaptive_grad_norm and grad_norm > grad_norm_threshold else "NORMAL"
                    progress_bar.set_postfix({
                        'loss': f'{reduced_loss.item():.4f}',
                        'avg_loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'grad_norm': f'{grad_norm:.3f}',
                        'status': grad_status,
                        'step': global_step
                    })
                    
                    # 记录到CSV
                    with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        grad_status = "ADAPTIVE" if adaptive_grad_norm and grad_norm > grad_norm_threshold else "NORMAL"
                        writer.writerow([
                            epoch+1,
                            global_step,
                            reduced_loss.item(),
                            avg_loss,
                            current_lr,
                            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            grad_status
                        ])
                
                # 保存检查点 (只在rank 0上执行)
                if rank == 0 and global_step % save_every == 0:
                    checkpoint_path = save_checkpoint(model, optimizer, epoch, global_step, reduced_loss.item())
                    print(f"\n检查点已保存: {checkpoint_path}")
                
                # 清理内存
                del loss, outputs, outputs_flat, targets_flat
                
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if rank == 0:
                    print(f"处理batch {batch_idx} 时出错: {e}")
                continue
        
        # 每个epoch结束 (只在rank 0上执行)
        if rank == 0 and valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            print(f"Epoch {epoch+1} 完成 | Avg Loss: {avg_loss:.4f} | Valid Batches: {valid_batches}/{len(train_loader)} | Total Steps: {global_step}")
            
            # 保存epoch检查点
            epoch_checkpoint = save_checkpoint(model, optimizer, epoch+1, global_step, avg_loss)
            print(f"Epoch检查点已保存: {epoch_checkpoint}")
        elif rank == 0:
            print(f"Epoch {epoch+1} 没有有效的batch")
        
        torch.cuda.empty_cache()
        dist.barrier()  # 确保所有进程同步
    
    # 最终保存 (只在rank 0上执行)
    if rank == 0:
        final_model_path = 'transformer_model_regularized.pt'
        
        # 保存原始模型 (去掉DDP包装)
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), final_model_path)
        else:
            torch.save(model.state_dict(), final_model_path)
        
        print(f"正则化训练完成，模型已保存为 {final_model_path}")
        
        # 保存完整的检查点
        final_checkpoint = save_checkpoint(model, optimizer, num_epochs, global_step, avg_loss if 'avg_loss' in locals() else 0)
        print(f"最终检查点已保存: {final_checkpoint}")

        print("标签平滑损失函数 (smoothing=0.1)")
        print(" 权重衰减正则化 (weight_decay=1e-4)")
        print(" 放宽梯度裁剪 (max_grad_norm=1.0)")
        print(" 使用原始论文学习率调度")
    
    # 清理分布式环境
    cleanup()

if __name__ == '__main__':
    world_size = 1  # 使用1个GPU
    
    # 使用torch.multiprocessing启动多进程
    torch.multiprocessing.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )