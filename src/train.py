# 修复 Windows 上的 OpenMP 冲突问题
# 必须在导入 torch 之前设置环境变量
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR
from tqdm import tqdm

# 兼容相对导入和绝对导入
# 如果以模块形式运行 (python -m src.train)，使用相对导入
# 如果直接运行 (python src/train.py)，使用绝对导入


try:
    from .data import load_dataset
    from .model import TransformerLM
    from .model import TransformerEncoderDecoder
    from .utils import TrainConfig, set_seed, get_device, ensure_dir, SmoothedValue, save_json, linear_warmup_cosine
except ImportError:
    # 添加项目根目录到 Python 路径，以便使用绝对导入
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.data import load_dataset
    from src.model import TransformerLM
    from src.model import TransformerEncoderDecoder
    from src.utils import TrainConfig, set_seed, get_device, ensure_dir, SmoothedValue, save_json, linear_warmup_cosine


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="tiny_shakespeare")
    p.add_argument("--vocab", type=str, default="byte")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_encoder_layers", type=int, default=4)
    p.add_argument("--n_decoder_layers", type=int, default=4)
    p.add_argument("--ffn_hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # 模型架构选项
    p.add_argument("--model_type", type=str, default="seq2seq", choices=["lm", "seq2seq"],
                   help="模型类型: 'lm' (语言模型) 或 'seq2seq' (编码器-解码器)")
    p.add_argument("--attention_type", type=str, default="linear", choices=["standard", "linear"],
                   help="注意力类型: 'standard' (标准) 或 'linear' (线性)")
    p.add_argument("--pos_encoding_type", type=str, default="relative", choices=["absolute", "relative"],
                   help="位置编码类型: 'absolute' (绝对) 或 'relative' (相对)")
    p.add_argument("--max_relative_position", type=int, default=128,
                   help="最大相对位置距离（仅用于相对位置编码）")

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--scheduler", type=str, default="cosine")
    p.add_argument("--warmup_steps", type=int, default=500)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--results_dir", type=str, default="results/exp_default")
    p.add_argument("--device", type=str, default=None)

    args = p.parse_args()
    cfg = TrainConfig(**vars(args))
    return cfg


def build_optimizer(params, name: str, lr: float, weight_decay: float):
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name}")


def build_scheduler(name: str, opt, max_steps: int, warmup: int):
    if name == "cosine":
        # 余弦 + 线性 warmup（手动实现，见训练循环中）
        return None
    if name == "onecycle":
        return OneCycleLR(opt, max_lr=opt.param_groups[0]['lr'], total_steps=max_steps)
    if name == "steplr":
        return StepLR(opt, step_size=max_steps // 3, gamma=0.5)
    if name == "none":
        return None
    raise ValueError(f"unknown scheduler {name}")

def evaluate(model: nn.Module, loader, device: torch.device, causal_mask=None, model_type: str = "lm") -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: 要评估的模型
        loader: 数据加载器
        device: 设备
        causal_mask: 预计算的因果掩码（可选，如果为 None 则每次重新构建）
        model_type: 模型类型，"lm" 或 "seq2seq"
    
    Returns:
        包含 loss 和 ppl 的字典
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    # 推荐使用 sum reduction（更安全），如果你想保留 mean，请用 batch_tokens 方式累加
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    # 只在 GPU 上使用 non_blocking
    use_non_blocking = device.type == "cuda"

    with torch.no_grad():
        for x, y in loader:
            # 根据设备类型选择传输方式
            if use_non_blocking:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)

            if model_type == "seq2seq":
                # 编码器-解码器模型
                tgt_input = y[:, :-1]
                tgt_output = y[:, 1:]
                tgt_mask = TransformerEncoderDecoder.build_causal_mask(tgt_input.size(1), device)
                logits = model(x, tgt_input, src_mask=None, tgt_causal_mask=tgt_mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
                batch_tokens = tgt_output.numel()
            else:
                # 语言模型
                T = x.size(1)
                if causal_mask is None:
                    mask = TransformerLM.build_causal_mask(T, device)
                else:
                    mask = causal_mask.to(device) if causal_mask.device != device else causal_mask
                    if mask.size(-1) != T:
                        mask = TransformerLM.build_causal_mask(T, device)

                logits = model(x, mask)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                batch_tokens = y.numel()

            total_loss += float(loss.item())   # loss is SUM over tokens in this batch
            total_tokens += batch_tokens

    avg_loss = total_loss / max(1, total_tokens)
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    return {"loss": avg_loss, "ppl": ppl}



def plot_curves(log_path: str, fig_path: str):
    """
    绘制训练曲线，包括 loss、perplexity 和 learning rate
    
    Args:
        log_path: 日志文件路径（JSON 格式）
        fig_path: 图片保存路径
    """
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)
    
    if len(logs) == 0:
        print("⚠ 警告: 日志为空，无法绘制曲线")
        return
    
    steps = [r["step"] for r in logs]
    train_loss = [r["train_loss"] for r in logs]
    eval_loss = [r["eval_loss"] for r in logs]
    eval_ppl = [r["eval_ppl"] for r in logs]
    lr = [r.get("lr", 0) for r in logs]  # 兼容可能没有 lr 的旧日志
    
    # 创建包含多个子图的图表
    fig = plt.figure(figsize=(14, 10))
    
    # 子图 1: Loss 曲线
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(steps, train_loss, 'b-', label='Train Loss', linewidth=1.5, alpha=0.8)
    ax1.plot(steps, eval_loss, 'r-', label='Eval Loss', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图 2: Perplexity 曲线
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(steps, eval_ppl, 'g-', label='Eval Perplexity', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Perplexity', fontsize=11)
    ax2.set_title('Validation Perplexity', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    # 使用对数刻度（如果 perplexity 较大）
    if max(eval_ppl) > 100:
        ax2.set_yscale('log')
    
    # 子图 3: Learning Rate 曲线
    ax3 = plt.subplot(2, 2, 3)
    if any(lr):  # 如果有学习率数据
        ax3.plot(steps, lr, 'm-', label='Learning Rate', linewidth=1.5, alpha=0.8)
        ax3.set_xlabel('Step', fontsize=11)
        ax3.set_ylabel('Learning Rate', fontsize=11)
        ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')  # 学习率通常使用对数刻度
    else:
        ax3.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    
    # 子图 4: Loss 差异（训练损失 - 验证损失）
    ax4 = plt.subplot(2, 2, 4)
    loss_diff = [tr - ev for tr, ev in zip(train_loss, eval_loss)]
    ax4.plot(steps, loss_diff, 'orange', label='Train Loss - Eval Loss', linewidth=1.5, alpha=0.8)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Loss Difference', fontsize=11)
    ax4.set_title('Overfitting Indicator', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    
    # 保存图片
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ 训练曲线已保存: {fig_path}")


def plot_existing_log(log_path: str, output_path: str = None):
    """
    绘制已有日志文件的训练曲线（用于训练后单独绘制）
    
    Args:
        log_path: 日志文件路径
        output_path: 输出图片路径（如果不指定，则使用日志文件同目录下的 curves.png）
    """
    if not os.path.exists(log_path):
        print(f"❌ 错误: 日志文件不存在: {log_path}")
        return
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(log_path), "curves.png")
    
    print(f"正在读取日志: {log_path}")
    plot_curves(log_path, output_path)
    print(f"✓ 图表已保存: {output_path}")


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    
    # 打印设备信息
    print(f"\n{'='*60}")
    print(f"训练配置")
    print(f"{'='*60}")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"序列长度: {cfg.seq_len}")
    print(f"模型参数: d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}")
    print(f"{'='*60}\n")

    ensure_dir(cfg.results_dir)

    # 优化 DataLoader：Windows 使用 0，Linux/Mac 可以使用多进程
    num_workers = 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)
    
    train_loader, val_loader, vocab_size = load_dataset(
        cfg.dataset, cfg.vocab, cfg.seq_len, cfg.batch_size, num_workers=num_workers
    )

    # 根据模型类型创建模型
    if cfg.model_type == "seq2seq":
        # from .model import TransformerEncoderDecoder
        model = TransformerEncoderDecoder(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            n_encoder_layers=cfg.n_encoder_layers,
            n_decoder_layers=cfg.n_decoder_layers,
            n_heads=cfg.n_heads,
            ffn_hidden=cfg.ffn_hidden,
            dropout=cfg.dropout,
            attention_type=cfg.attention_type,
            pos_encoding_type=cfg.pos_encoding_type,
            max_relative_position=cfg.max_relative_position,
        ).to(device)
    else:
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_hidden=cfg.ffn_hidden,
            dropout=cfg.dropout,
            attention_type=cfg.attention_type,
            pos_encoding_type=cfg.pos_encoding_type,
            max_relative_position=cfg.max_relative_position,
        ).to(device)
    
    # 打印模型配置信息
    print(f"模型类型: {cfg.model_type}")
    print(f"注意力类型: {cfg.attention_type}")
    print(f"位置编码类型: {cfg.pos_encoding_type}")
    if cfg.pos_encoding_type == "relative":
        print(f"最大相对位置: {cfg.max_relative_position}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,}\n")

    optimizer = build_optimizer(model.parameters(), cfg.optimizer, cfg.lr, cfg.weight_decay)
    scheduler = build_scheduler(cfg.scheduler, optimizer, cfg.max_steps, cfg.warmup_steps)

    loss_fn = nn.CrossEntropyLoss()

    # 预计算因果掩码（序列长度固定时，仅用于语言模型）
    if cfg.model_type == "lm":
        causal_mask = model.build_causal_mask(cfg.seq_len, device)
        print(f"✓ 预计算因果掩码 (序列长度={cfg.seq_len})\n")
    else:
        causal_mask = None  # 编码器-解码器模型会在前向传播中构建掩码
        print(f"✓ 编码器-解码器模式（掩码在训练时动态构建）\n")

    log: list = []
    best_eval = float("inf")
    step = 0
    
    # 性能统计
    start_time = time.time()
    last_log_time = start_time
    
    pbar = tqdm(total=cfg.max_steps, desc="training", unit="step")

    # 只在 GPU 上使用 non_blocking
    use_non_blocking = device.type == "cuda"
    
    while step < cfg.max_steps:
        for x, y in train_loader:
            if step >= cfg.max_steps:
                break
            model.train()
            
            # 根据设备类型选择传输方式
            if use_non_blocking:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)
            
            # 根据模型类型进行前向传播
            if cfg.model_type == "seq2seq":
                # 编码器-解码器：输入 x 作为编码器输入，y 的前 n-1 个 token 作为解码器输入
                # y[:-1] 作为解码器输入，y[1:] 作为目标
                tgt_input = y[:, :-1]  # [B, T-1]
                tgt_output = y[:, 1:]   # [B, T-1]
                tgt_mask = model.build_causal_mask(tgt_input.size(1), device) if hasattr(model, 'build_causal_mask') else None
                # 注意：这里简化处理，实际 seq2seq 任务需要不同的输入格式
                # 当前假设编码器和解码器使用相同输入（用于测试）
                logits = model(x, tgt_input, src_mask=None, tgt_causal_mask=tgt_mask)
                # 计算损失时只考虑解码器输出部分
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            else:
                # 语言模型：使用预计算的掩码
                mask = causal_mask.to(device) if causal_mask.device != device else causal_mask
                logits = model(x, mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 手动 warmup+cosine（独立于torch scheduler，若选择cosine）
            if cfg.scheduler == "cosine":
                scale = linear_warmup_cosine(step, cfg.warmup_steps, cfg.max_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = cfg.lr * scale
            elif scheduler is not None:
                scheduler.step()

            step += 1
            
            # 性能统计：每 100 步更新一次速度信息
            if step % 100 == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                # 防止除以零
                if elapsed > 0:
                    steps_per_sec = 100 / elapsed
                else:
                    steps_per_sec = 0
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "speed": f"{steps_per_sec:.1f} step/s",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                last_log_time = current_time
            
            if step % cfg.eval_interval == 0 or step == 1:
                # 评估模型
                eval_start = time.time()
                ev = evaluate(model, val_loader, device, causal_mask, cfg.model_type)
                eval_time = time.time() - eval_start
                
                rec = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "eval_loss": ev["loss"],
                    "eval_ppl": ev["ppl"],
                    "lr": optimizer.param_groups[0]['lr'],
                }
                log.append(rec)
                
                # 异步保存（减少 IO 阻塞）
                log_path = os.path.join(cfg.results_dir, "train_log.json")
                save_json(log, log_path)
                
                # 保存模型
                if ev["loss"] < best_eval:
                    best_eval = ev["loss"]
                    torch.save(model.state_dict(), os.path.join(cfg.results_dir, "model_best.pt"))
                torch.save(model.state_dict(), os.path.join(cfg.results_dir, "model_last.pt"))
                
                # 更新曲线（仅在需要时）
                plot_curves(log_path, os.path.join(cfg.results_dir, "curves.png"))
                
                # 打印评估信息
                elapsed_total = time.time() - start_time
                pbar.write(
                    f"Step {step}: train_loss={loss.item():.4f}, "
                    f"eval_loss={ev['loss']:.4f}, eval_ppl={ev['ppl']:.2f}, "
                    f"eval_time={eval_time:.2f}s, total_time={elapsed_total/60:.1f}min"
                )
            
            pbar.update(1)
            if step >= cfg.max_steps:
                break

    pbar.close()
    
    # 训练结束时绘制最终图表
    log_path = os.path.join(cfg.results_dir, "train_log.json")
    if os.path.exists(log_path):
        print("\n正在生成最终训练曲线...")
        plot_curves(log_path, os.path.join(cfg.results_dir, "curves.png"))
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ 训练完成！")
    print(f"{'='*60}")
    print(f"总用时: {total_time/60:.1f} 分钟 ({total_time:.1f} 秒)")
    print(f"平均速度: {cfg.max_steps / total_time:.2f} step/s")
    if log:
        final_loss = log[-1]
        print(f"最终训练损失: {final_loss['train_loss']:.4f}")
        print(f"最终验证损失: {final_loss['eval_loss']:.4f}")
        print(f"最终困惑度: {final_loss['eval_ppl']:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    # 如果提供了 --plot 参数，则只绘制图表
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        if len(sys.argv) < 3:
            print("用法: python -m src.train --plot <日志文件路径> [输出图片路径]")
            print("示例: python -m src.train --plot results/exp_default/train_log.json")
            sys.exit(1)
        log_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        plot_existing_log(log_file, output_file)
    else:
        main()

