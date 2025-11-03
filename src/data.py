import io
import os
from typing import Tuple

import requests
import torch
from torch.utils.data import Dataset, DataLoader


_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class CharByteTokenizer:
    def __init__(self, mode: str = "byte"):
        assert mode in {"byte", "char"}
        self.mode = mode
        if mode == "byte":
            self.vocab_size = 256
        else:
            # 字符级将基于数据动态构建词表
            self.stoi = {}
            self.itos = {}
            self.vocab_size = 0

    def build_for_text(self, text: str):
        if self.mode == "char":
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(chars)

    def encode(self, text: str) -> torch.Tensor:
        if self.mode == "byte":
            data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        else:
            data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        return data

    def decode(self, ids: torch.Tensor) -> str:
        if self.mode == "byte":
            return bytes([int(x) for x in ids]).decode("utf-8", errors="ignore")
        return "".join(self.itos[int(i)] for i in ids)


class LMSequenceDataset(Dataset):
    def __init__(self, ids: torch.Tensor, seq_len: int):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + 1 + self.seq_len]
        return x, y


def _download_tiny_shakespeare(cache_dir: str) -> str:
    """
    下载 Tiny Shakespeare 数据集
    
    Args:
        cache_dir: 缓存目录
        
    Returns:
        数据集文件路径
        
    Raises:
        Exception: 如果下载失败
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "tinyshakespeare.txt")
    
    # 如果文件已存在，直接返回
    if os.path.exists(path):
        file_size = os.path.getsize(path)
        if file_size > 0:  # 检查文件是否非空
            print(f"✓ 使用已缓存的数据集: {path} ({file_size / 1024:.1f} KB)")
            return path
        else:
            print(f"⚠ 缓存文件为空，重新下载...")
            os.remove(path)
    
    # 下载数据集
    print(f"正在从 GitHub 下载 Tiny Shakespeare 数据集...")
    print(f"URL: {_TINY_SHAKESPEARE_URL}")
    try:
        r = requests.get(_TINY_SHAKESPEARE_URL, timeout=60)
        r.raise_for_status()
        
        # 保存文件
        with open(path, "wb") as f:
            f.write(r.content)
        
        file_size = os.path.getsize(path)
        print(f"✓ 数据集下载完成: {path} ({file_size / 1024:.1f} KB)")
        return path
        
    except requests.exceptions.RequestException as e:
        error_msg = (
            f"❌ 下载数据集失败！\n"
            f"错误信息: {str(e)}\n"
            f"可能原因:\n"
            f"  1. 网络连接问题（无法访问 GitHub）\n"
            f"  2. 代理设置问题\n"
            f"  3. 防火墙阻止\n\n"
            f"解决方案:\n"
            f"  1. 检查网络连接\n"
            f"  2. 手动下载数据集并放置到: {path}\n"
            f"     下载地址: {_TINY_SHAKESPEARE_URL}\n"
            f"  3. 或使用代理: export https_proxy=your_proxy"
        )
        raise RuntimeError(error_msg) from e


def load_dataset(name: str, vocab: str, seq_len: int, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, int]:
    """
    加载数据集并创建数据加载器
    
    Args:
        name: 数据集名称（目前仅支持 "tiny_shakespeare"）
        vocab: 词汇表类型（"byte" 或 "char"）
        seq_len: 序列长度
        batch_size: 批次大小
        num_workers: DataLoader 的工作进程数（Windows 建议设为 0）
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        vocab_size: 词汇表大小
    """
    assert name in {"tiny_shakespeare"}, f"不支持的数据集: {name}。当前仅支持: tiny_shakespeare"
    assert vocab in {"byte", "char"}, f"不支持的词汇表类型: {vocab}。支持: byte, char"

    print(f"\n{'='*60}")
    print(f"加载数据集: {name}")
    print(f"词汇表类型: {vocab}, 序列长度: {seq_len}, 批次大小: {batch_size}")
    print(f"{'='*60}\n")

    cache_dir = os.path.join(".cache", name)
    
    try:
        # 下载或获取数据集路径
        txt_path = _download_tiny_shakespeare(cache_dir)
        
        # 读取文本文件
        print(f"正在读取数据集文件...")
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        if len(text) == 0:
            raise ValueError(f"数据集文件为空: {txt_path}")
        
        print(f"✓ 数据集读取完成，总字符数: {len(text):,}")
        
    except Exception as e:
        error_msg = (
            f"\n❌ 加载数据集失败！\n"
            f"错误: {str(e)}\n\n"
            f"请确保:\n"
            f"  1. 网络连接正常（首次运行需要下载数据集）\n"
            f"  2. 有足够的磁盘空间\n"
            f"  3. 如果持续失败，请手动下载数据集到: {os.path.join('.cache', name, 'tinyshakespeare.txt')}\n"
        )
        raise RuntimeError(error_msg) from e

    # 数据集切分：90% 训练，10% 验证
    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9) :]
    
    print(f"训练集: {len(train_text):,} 字符")
    print(f"验证集: {len(val_text):,} 字符")

    # 构建分词器
    print(f"\n构建分词器 (mode={vocab})...")
    tok = CharByteTokenizer(vocab)
    tok.build_for_text(train_text)
    print(f"✓ 词汇表大小: {tok.vocab_size}")

    # 编码文本为 token ID
    print(f"编码文本...")
    train_ids = tok.encode(train_text)
    val_ids = tok.encode(val_text)
    print(f"✓ 训练集 tokens: {len(train_ids):,}")
    print(f"✓ 验证集 tokens: {len(val_ids):,}")

    # 创建数据集
    train_ds = LMSequenceDataset(train_ids, seq_len)
    val_ds = LMSequenceDataset(val_ids, seq_len)
    
    print(f"\n创建数据加载器...")
    print(f"训练样本数: {len(train_ds):,}")
    print(f"验证样本数: {len(val_ds):,}")

    def collate(batch):
        """批处理函数：将多个样本堆叠成批次"""
        xs, ys = zip(*batch)
        return torch.stack(xs, 0), torch.stack(ys, 0)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate
    )

    vocab_size = tok.vocab_size
    print(f"\n✓ 数据集加载完成！词汇表大小: {vocab_size}\n")
    return train_loader, val_loader, vocab_size

