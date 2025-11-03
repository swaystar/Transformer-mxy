"""
Transformer 模型实现
包含位置编码、多头自注意力、Transformer 块和完整的语言模型
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    绝对位置编码（Absolute Positional Encoding）模块
    
    Transformer 本身无法感知序列中 token 的位置信息，因此需要通过位置编码来注入位置信息。
    这里实现的是原始论文中的正弦/余弦位置编码：
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中 pos 是位置，i 是维度索引。
    """
    
    def __init__(self, d_model: int, max_len: int = 8192):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度（嵌入维度）
            max_len: 支持的最大序列长度
        """
        super().__init__()
        # 创建位置编码矩阵：[max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引 [0, 1, 2, ..., max_len-1]，形状为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除数项：10000^(2i/d_model)，用于不同维度使用不同的频率
        # div_term 形状为 [d_model // 2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # 偶数维度使用正弦：PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度使用余弦：PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将位置编码注册为 buffer（不参与梯度更新），形状扩展为 [1, max_len, d_model]
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将位置编码添加到输入嵌入中
        
        Args:
            x: 输入张量，形状为 [B, T, d_model]
               B: batch size（批次大小）
               T: 序列长度（token 数量）
        
        Returns:
            添加位置编码后的张量，形状为 [B, T, d_model]
        """
        T = x.size(1)
        # 将位置编码与输入嵌入相加（广播机制）
        return x + self.pe[:, :T, :].to(x.dtype)


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码（Relative Positional Encoding）模块
    
    相对位置编码通过编码 token 之间的相对位置关系，而不是绝对位置。
    这里实现的是 Shaw et al. (2018) 的方法，将相对位置信息融入到注意力计算中。
    
    相比绝对位置编码，相对位置编码的优势：
    1. 对长序列的泛化能力更强
    2. 更好地处理不同长度的序列
    """
    
    def __init__(self, d_model: int, max_len: int = 512, max_relative_position: int = 128):
        """
        初始化相对位置编码
        
        Args:
            d_model: 模型维度
            max_len: 支持的最大序列长度
            max_relative_position: 最大相对位置距离（超过此距离的相对位置编码相同）
        """
        super().__init__()
        self.max_relative_position = max_relative_position
        # 相对位置嵌入：每个相对位置（从 -max_relative_position 到 max_relative_position）都有一个嵌入
        # 形状: [2 * max_relative_position + 1, d_model]
        self.relative_pos_emb = nn.Embedding(2 * max_relative_position + 1, d_model)
        
        # 生成相对位置索引矩阵
        # 对于位置 i 和 j，相对位置为 j - i
        position_range = torch.arange(-max_relative_position, max_relative_position + 1)
        # 将超出范围的位置映射到边界
        position_range = torch.clamp(position_range, -max_relative_position, max_relative_position)
        # 转换为嵌入索引（从 0 开始）
        self.register_buffer("position_range", position_range + max_relative_position, persistent=False)
    
    def get_relative_position_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        获取相对位置矩阵
        
        Args:
            seq_len: 序列长度
            device: 设备
        
        Returns:
            相对位置索引矩阵，形状为 [seq_len, seq_len]
            矩阵[i, j] 表示位置 j 相对于位置 i 的相对位置索引
        """
        # 创建相对位置矩阵：rel_pos[i, j] = j - i
        positions = torch.arange(seq_len, device=device).unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)
        # 将相对位置限制在 [-max_relative_position, max_relative_position] 范围内
        positions = torch.clamp(positions, -self.max_relative_position, self.max_relative_position)
        # 转换为嵌入索引（从 0 开始）
        positions = positions + self.max_relative_position
        return positions
    
    def forward(self, q: torch.Tensor, k: torch.Tensor = None) -> torch.Tensor:
        """
        获取相对位置编码
        
        Args:
            q: 查询张量，形状为 [B, n_heads, T, head_dim]
            k: 键张量（可选），形状为 [B, n_heads, T, head_dim]
               如果为 None，则假设与 q 相同（自注意力）
        
        Returns:
            相对位置编码，形状为 [B, n_heads, T, T, head_dim]
        """
        B, n_heads, T, head_dim = q.size()
        
        # 获取相对位置矩阵
        rel_pos_matrix = self.get_relative_position_matrix(T, q.device)  # [T, T]
        # 获取相对位置嵌入
        rel_pos_emb = self.relative_pos_emb(rel_pos_matrix)  # [T, T, d_model]
        
        # 将相对位置嵌入投影到每个头的维度
        # 这里简化处理：将 d_model 维的嵌入平均分配到各个头
        # 实际实现中可能需要单独的参数
        rel_pos_emb = rel_pos_emb.view(T, T, n_heads, head_dim)  # [T, T, n_heads, head_dim]
        rel_pos_emb = rel_pos_emb.permute(2, 0, 1, 3)  # [n_heads, T, T, head_dim]
        rel_pos_emb = rel_pos_emb.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, n_heads, T, T, head_dim]
        
        return rel_pos_emb


class LinearAttention(nn.Module):
    """
    线性注意力（Linear Attention）机制
    
    标准注意力的复杂度是 O(n^2)，其中 n 是序列长度。
    线性注意力通过使用特征映射 φ 将复杂度降低到 O(n)：
    
    Linear-Attention(Q, K, V) = φ(Q) @ (φ(K)^T @ V)
    
    其中 φ 是特征映射函数，通常是 elu(x) + 1 或其他正的特征函数。
    
    参考：Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        """
        初始化线性注意力模块
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数量
            dropout: Dropout 比率
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q、K、V 投影
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征映射函数 φ(x)
        
        使用 elu(x) + 1 作为特征映射，确保输出为正且可导
        """
        return F.elu(x) + 1.0
    
    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：计算线性注意力
        
        Args:
            x: 输入张量，形状为 [B, T, d_model]
            causal_mask: 因果掩码（可选）
        
        Returns:
            注意力输出，形状为 [B, T, d_model]
        """
        B, T, C = x.size()
        
        # 计算 Q、K、V
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状: [B, n_heads, T, head_dim]
        
        # 应用特征映射
        q_mapped = self.feature_map(q / (self.head_dim ** 0.25))  # [B, n_heads, T, head_dim]
        k_mapped = self.feature_map(k / (self.head_dim ** 0.25))  # [B, n_heads, T, head_dim]
        
        # 线性注意力：φ(Q) @ (φ(K)^T @ V)
        # 先计算 φ(K)^T @ V，形状为 [B, n_heads, head_dim, head_dim]
        kv = torch.einsum('bhnd,bhnm->bhdm', k_mapped, v)  # [B, n_heads, head_dim, head_dim]
        
        # 计算 φ(Q) @ kv，形状为 [B, n_heads, T, head_dim]
        qkv_out = torch.einsum('bhnd,bhdm->bhnm', q_mapped, kv)  # [B, n_heads, T, head_dim]
        
        # 归一化：除以每个查询的总和
        z = q_mapped.sum(dim=-1, keepdim=True)  # [B, n_heads, T, 1]
        qkv_out = qkv_out / (z + 1e-6)  # 避免除以零
        
        # 合并多头
        qkv_out = qkv_out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        out = self.dropout(self.out(qkv_out))
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention）模块
    
    这是 Transformer 的核心组件，通过并行计算多个注意力头来捕获不同类型的关系。
    实现的是缩放点积注意力（Scaled Dot-Product Attention）：
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    其中 Q（查询）、K（键）、V（值）都来自同一个输入，因此是"自注意力"。
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_relative_pos: bool = False, relative_pos_emb: Optional[nn.Module] = None):
        """
        初始化多头自注意力模块
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头的数量
            dropout: Dropout 比率
            use_relative_pos: 是否使用相对位置编码
            relative_pos_emb: 相对位置编码模块（如果使用）
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 每个头的维度
        self.use_relative_pos = use_relative_pos
        self.relative_pos_emb = relative_pos_emb
        
        # 使用单个线性层同时生成 Q、K、V（更高效）
        # 输出维度为 d_model * 3，分别对应 Q、K、V
        self.qkv = nn.Linear(d_model, d_model * 3)
        # 输出投影层，将多头注意力结果合并
        self.out = nn.Linear(d_model, d_model)
        # 注意力权重的 Dropout
        self.attn_drop = nn.Dropout(dropout)
        # 残差连接的 Dropout
        self.resid_drop = nn.Dropout(dropout)
        
        # 相对位置编码需要额外的参数（用于将相对位置信息融入到注意力计算）
        if use_relative_pos:
            # 用于将相对位置信息添加到注意力分数中的投影
            self.rel_pos_proj_q = nn.Linear(d_model, d_model, bias=False)
            self.rel_pos_proj_k = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：计算多头自注意力
        
        Args:
            x: 输入张量，形状为 [B, T, d_model]
            causal_mask: 因果掩码（用于语言模型，防止看到未来信息），形状为 [1, 1, T, T]
                        None 表示不使用掩码（用于双向编码）
        
        Returns:
            注意力输出，形状为 [B, T, d_model]
        """
        B, T, C = x.size()
        
        # 计算 Q、K、V
        # qkv: [B, T, d_model * 3]
        qkv = self.qkv(x)
        # 重塑为 [B, T, 3, n_heads, head_dim]
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        # 重排列为 [3, B, n_heads, T, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # 分离 Q、K、V，每个形状为 [B, n_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数：Q @ K^T，形状为 [B, n_heads, T, T]
        # 除以 sqrt(head_dim) 进行缩放，防止点积值过大导致 softmax 饱和
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 如果使用相对位置编码，添加到注意力分数中
        if self.use_relative_pos and self.relative_pos_emb is not None:
            # 获取相对位置编码
            rel_pos_emb = self.relative_pos_emb(q, k)  # [B, n_heads, T, T, head_dim]
            # 简化：将相对位置编码添加到注意力分数
            # 这里简化处理，实际实现可能需要更复杂的计算
            rel_pos_bias = (rel_pos_emb.sum(dim=-1) / self.head_dim ** 0.5)  # [B, n_heads, T, T]
            att = att + rel_pos_bias
        
        # 应用因果掩码（用于自回归语言模型）
        # 掩码位置设为 -inf，softmax 后权重为 0
        if causal_mask is not None:
            att = att.masked_fill(causal_mask == 0, float("-inf"))
        
        # 应用 softmax 得到注意力权重，形状为 [B, n_heads, T, T]
        att = F.softmax(att, dim=-1)
        # Dropout 正则化
        att = self.attn_drop(att)
        
        # 加权求和：注意力权重 @ V，形状为 [B, n_heads, T, head_dim]
        y = att @ v
        
        # 合并多头：将多个头的输出拼接
        # 先转置为 [B, T, n_heads, head_dim]
        y = y.transpose(1, 2).contiguous()
        # 重塑为 [B, T, d_model]
        y = y.view(B, T, C)
        
        # 输出投影
        y = self.resid_drop(self.out(y))
        return y


class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力（Multi-Head Cross-Attention）模块
    
    交叉注意力用于解码器中，从编码器输出（encoder_output）中获取信息。
    查询（Q）来自解码器，键（K）和值（V）来自编码器。
    
    Cross-Attention(Q_dec, K_enc, V_enc) = softmax(Q_dec @ K_enc^T / sqrt(d_k)) @ V_enc
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        """
        初始化多头交叉注意力模块
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数量
            dropout: Dropout 比率
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q 来自解码器输入
        self.q_proj = nn.Linear(d_model, d_model)
        # K 和 V 来自编码器输出
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：计算交叉注意力
        
        Args:
            x: 解码器输入，形状为 [B, T_dec, d_model]
            encoder_output: 编码器输出，形状为 [B, T_enc, d_model]
            mask: 编码器-解码器掩码（可选），形状为 [1, 1, T_dec, T_enc]
        
        Returns:
            交叉注意力输出，形状为 [B, T_dec, d_model]
        """
        B, T_dec, C = x.size()
        T_enc = encoder_output.size(1)
        
        # 计算 Q（来自解码器）
        q = self.q_proj(x)  # [B, T_dec, d_model]
        q = q.view(B, T_dec, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, n_heads, T_dec, head_dim]
        
        # 计算 K、V（来自编码器）
        kv = self.kv_proj(encoder_output)  # [B, T_enc, d_model * 2]
        kv = kv.view(B, T_enc, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # 每个形状: [B, n_heads, T_enc, head_dim]
        
        # 计算注意力分数：Q @ K^T，形状为 [B, n_heads, T_dec, T_enc]
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用掩码（如果提供）
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        
        # Softmax 和 Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # 加权求和
        y = att @ v  # [B, n_heads, T_dec, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T_dec, C)
        
        # 输出投影
        y = self.resid_drop(self.out(y))
        return y


class TransformerBlock(nn.Module):
    """
    Transformer 编码器块（Encoder Block）
    
    完整的 Transformer 块包含：
    1. 多头自注意力层（带残差连接和层归一化）
    2. 位置前馈网络（Position-wise Feed-Forward Network，带残差连接和层归一化）
    
    这里使用 Pre-LayerNorm 结构（LayerNorm 在子层之前），相比 Post-LayerNorm 更稳定。
    """
    
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float, 
                 attention_type: str = "standard", use_relative_pos: bool = False, 
                 relative_pos_emb: Optional[nn.Module] = None):
        """
        初始化 Transformer 块
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数量
            ffn_hidden: 前馈网络的隐藏层维度（通常是 d_model 的 4 倍）
            dropout: Dropout 比率
            attention_type: 注意力类型，"standard" 或 "linear"
            use_relative_pos: 是否使用相对位置编码
            relative_pos_emb: 相对位置编码模块（如果使用）
        """
        super().__init__()
        # 第一个 LayerNorm（用于注意力层之前）
        self.ln1 = nn.LayerNorm(d_model)
        
        # 根据类型选择注意力机制
        if attention_type == "linear":
            self.attn = LinearAttention(d_model, n_heads, dropout)
        else:
            self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, use_relative_pos, relative_pos_emb)
        
        # 第二个 LayerNorm（用于前馈网络之前）
        self.ln2 = nn.LayerNorm(d_model)
        # 位置前馈网络（FFN）
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        使用 Pre-LayerNorm 结构：
        1. x -> LayerNorm -> Attention -> + x (残差连接)
        2. x -> LayerNorm -> FFN -> + x (残差连接)
        
        Args:
            x: 输入张量，形状为 [B, T, d_model]
            causal_mask: 因果掩码（传递给注意力层）
        
        Returns:
            输出张量，形状为 [B, T, d_model]
        """
        # 第一个子层：多头自注意力 + 残差连接
        x = x + self.attn(self.ln1(x), causal_mask)
        
        # 第二个子层：前馈网络 + 残差连接
        x = x + self.ff(self.ln2(x))
        
        return x


class DecoderBlock(nn.Module):
    """
    Transformer 解码器块（Decoder Block）
    
    解码器块包含三个子层：
    1. 自注意力层（带因果掩码）
    2. 交叉注意力层（从编码器获取信息）
    3. 前馈网络层
    
    每个子层都有残差连接和层归一化。
    """
    
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float,
                 attention_type: str = "standard", use_relative_pos: bool = False,
                 relative_pos_emb: Optional[nn.Module] = None):
        """
        初始化解码器块
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数量
            ffn_hidden: 前馈网络隐藏层维度
            dropout: Dropout 比率
            attention_type: 注意力类型，"standard" 或 "linear"
            use_relative_pos: 是否使用相对位置编码
            relative_pos_emb: 相对位置编码模块
        """
        super().__init__()
        
        # 第一个子层：自注意力（带因果掩码）
        self.ln1 = nn.LayerNorm(d_model)
        if attention_type == "linear":
            self.self_attn = LinearAttention(d_model, n_heads, dropout)
        else:
            self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout, use_relative_pos, relative_pos_emb)
        
        # 第二个子层：交叉注意力
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        
        # 第三个子层：前馈网络
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                causal_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 解码器输入，形状为 [B, T_dec, d_model]
            encoder_output: 编码器输出，形状为 [B, T_enc, d_model]
            causal_mask: 因果掩码（用于自注意力）
            cross_mask: 交叉注意力掩码（可选）
        
        Returns:
            输出张量，形状为 [B, T_dec, d_model]
        """
        # 子层 1：自注意力
        x = x + self.self_attn(self.ln1(x), causal_mask)
        
        # 子层 2：交叉注意力
        x = x + self.cross_attn(self.ln2(x), encoder_output, cross_mask)
        
        # 子层 3：前馈网络
        x = x + self.ff(self.ln3(x))
        
        return x


class TransformerEncoderDecoder(nn.Module):
    """
    Transformer 编码器-解码器模型（Encoder-Decoder Transformer）
    
    完整的序列到序列（Seq2Seq）模型，包含：
    1. 编码器（Encoder）：处理输入序列
    2. 解码器（Decoder）：生成输出序列
    
    适用于机器翻译、文本摘要等任务。
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_encoder_layers: int, n_decoder_layers: int,
                 n_heads: int, ffn_hidden: int, dropout: float, 
                 attention_type: str = "standard", pos_encoding_type: str = "absolute",
                 max_relative_position: int = 128):
        """
        初始化编码器-解码器模型
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            n_encoder_layers: 编码器层数
            n_decoder_layers: 解码器层数
            n_heads: 注意力头数量
            ffn_hidden: 前馈网络隐藏层维度
            dropout: Dropout 比率
            attention_type: 注意力类型，"standard" 或 "linear"
            pos_encoding_type: 位置编码类型，"absolute" 或 "relative"
            max_relative_position: 最大相对位置距离
        """
        super().__init__()
        self.d_model = d_model
        self.attention_type = attention_type
        self.pos_encoding_type = pos_encoding_type
        
        # 嵌入层
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        if pos_encoding_type == "relative":
            self.src_pos_enc = None  # 相对位置编码在注意力中处理
            self.tgt_pos_enc = None
            self.relative_pos_emb = RelativePositionalEncoding(d_model, max_relative_position=max_relative_position)
        else:
            self.src_pos_enc = PositionalEncoding(d_model)
            self.tgt_pos_enc = PositionalEncoding(d_model)
            self.relative_pos_emb = None
        
        use_relative_pos = (pos_encoding_type == "relative")
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_hidden, dropout, attention_type, use_relative_pos, self.relative_pos_emb)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, ffn_hidden, dropout, attention_type, use_relative_pos, self.relative_pos_emb)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # 输出投影
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        """权重初始化"""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                tgt_causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 源序列（编码器输入），形状为 [B, T_src]
            tgt: 目标序列（解码器输入），形状为 [B, T_tgt]
            src_mask: 源序列掩码（可选）
            tgt_causal_mask: 目标序列因果掩码（可选）
        
        Returns:
            输出 logits，形状为 [B, T_tgt, vocab_size]
        """
        # 编码器
        src_emb = self.src_emb(src)
        if self.src_pos_enc is not None:
            src_emb = self.src_pos_enc(src_emb)
        
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, causal_mask=None)  # 编码器不需要因果掩码
        encoder_output = self.encoder_norm(encoder_output)
        
        # 解码器
        tgt_emb = self.tgt_emb(tgt)
        if self.tgt_pos_enc is not None:
            tgt_emb = self.tgt_pos_enc(tgt_emb)
        
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_causal_mask, src_mask)
        decoder_output = self.decoder_norm(decoder_output)
        
        # 输出投影
        logits = self.lm_head(decoder_output)
        return logits
    
    @staticmethod
    def build_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """构建因果掩码（与 TransformerLM 相同）"""
        mask = torch.tril(torch.ones(size, size, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask


class TransformerLM(nn.Module):
    """
    Transformer 语言模型（Language Model）
    
    一个完整的自回归语言模型，用于预测序列中下一个 token 的概率。
    模型结构：
    1. Token 嵌入层
    2. 位置编码
    3. N 个 Transformer 块（堆叠）
    4. 最终的层归一化
    5. 语言模型头（输出词汇表大小的 logits）
    
    支持绝对/相对位置编码和标准/线性注意力。
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, ffn_hidden: int, dropout: float,
                 attention_type: str = "standard", pos_encoding_type: str = "absolute", 
                 max_relative_position: int = 128):
        """
        初始化 Transformer 语言模型
        
        Args:
            vocab_size: 词汇表大小（token 的种类数）
            d_model: 模型维度（嵌入维度）
            n_layers: Transformer 块的层数
            n_heads: 每个块中注意力头的数量
            ffn_hidden: 前馈网络的隐藏层维度
            dropout: Dropout 比率
            attention_type: 注意力类型，"standard" 或 "linear"
            pos_encoding_type: 位置编码类型，"absolute" 或 "relative"
            max_relative_position: 最大相对位置距离（仅用于相对位置编码）
        """
        super().__init__()
        self.attention_type = attention_type
        self.pos_encoding_type = pos_encoding_type
        
        # Token 嵌入层：将 token ID 转换为 d_model 维的向量
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # 位置编码：为序列添加位置信息
        if pos_encoding_type == "relative":
            self.pos_enc = None  # 相对位置编码在注意力中处理
            self.relative_pos_emb = RelativePositionalEncoding(d_model, max_relative_position=max_relative_position)
        else:
            self.pos_enc = PositionalEncoding(d_model)
            self.relative_pos_emb = None
        
        use_relative_pos = (pos_encoding_type == "relative")
        
        # 堆叠多个 Transformer 块
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_hidden, dropout, attention_type, use_relative_pos, self.relative_pos_emb)
            for _ in range(n_layers)
        ])
        
        # 最终的层归一化（在所有 Transformer 块之后）
        self.ln_f = nn.LayerNorm(d_model)
        
        # 语言模型头：将隐藏状态投影到词汇表大小
        # 不使用偏置（bias=False），这是常见的做法
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 初始化权重
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """
        权重初始化函数
        
        对线性层和嵌入层使用正态分布初始化（均值 0，标准差 0.02），
        这是 Transformer 模型的常见初始化方法。
        """
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # 如果有偏置，初始化为 0
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        流程：
        1. Token 嵌入：[B, T] -> [B, T, d_model]
        2. 添加位置编码：[B, T, d_model] -> [B, T, d_model]（如果使用绝对位置编码）
        3. 通过 N 个 Transformer 块（如果使用相对位置编码，在注意力中处理）
        4. 最终层归一化
        5. 投影到词汇表：-> [B, T, vocab_size]
        
        Args:
            idx: Token ID 序列，形状为 [B, T]
                 B: batch size（批次大小）
                 T: 序列长度（token 数量）
            causal_mask: 因果掩码，如果为 None 则自动构建
        
        Returns:
            每个位置的下一个 token 的 logits，形状为 [B, T, vocab_size]
        """
        # 1. Token 嵌入：将 token ID 转换为嵌入向量
        x = self.tok_emb(idx)
        
        # 2. 添加位置编码（如果使用绝对位置编码）
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        
        # 3. 通过所有 Transformer 块
        for blk in self.layers:
            x = blk(x, causal_mask)
        
        # 4. 最终层归一化
        x = self.ln_f(x)
        
        # 5. 投影到词汇表大小，得到 logits（未归一化的分数）
        logits = self.lm_head(x)
        return logits

    @staticmethod
    def build_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """
        构建因果掩码（Causal Mask）
        
        用于自回归语言模型，确保在预测位置 i 的 token 时，
        只能看到位置 0 到 i-1 的信息，不能看到未来的信息。
        
        掩码矩阵是下三角矩阵：
        [[1, 0, 0, ...],
         [1, 1, 0, ...],
         [1, 1, 1, ...],
         ...]
        
        Args:
            size: 序列长度（T）
            device: 设备（CPU/GPU）
        
        Returns:
            因果掩码，形状为 [1, 1, T, T]
            1 表示保留，0 表示掩码（会被设为 -inf）
        """
        # 创建下三角矩阵（torch.tril）
        # 形状为 [T, T]
        mask = torch.tril(torch.ones(size, size, device=device))
        # 扩展维度以便广播，形状为 [1, 1, T, T]
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

