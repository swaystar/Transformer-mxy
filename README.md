# Transformer 作业实现

本仓库实现了一个从零搭建的基于 PyTorch 的小规模 Transformer 模型，支持：
- ✅ **语言模型（Language Model）**：自回归生成
- ✅ **编码器-解码器（Encoder-Decoder）**：序列到序列任务
- ✅ **多头注意力**：标准注意力 + 线性注意力（O(n) 复杂度）
- ✅ **位置编码**：绝对位置编码 + 相对位置编码
- ✅ **完整训练流程**：学习率调度、梯度裁剪、AdamW、模型保存、训练曲线可视化

---

## 📋 目录结构

```
Transformer/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖包
├── Description_of_the_Assignment.pdf  # 作业要求文档
├── src/                         # 源代码目录
│   ├── __init__.py             # 包初始化
│   ├── model.py                # Transformer 模型实现（833行）
│   ├── data.py                 # 数据集加载和预处理（223行）
│   ├── train.py                # 训练和验证脚本（500行）
│   └── utils.py                 # 配置类和工具函数（97行）
├── scripts/                     # 运行脚本
│   ├── run.sh                  # Bash 脚本（Linux/macOS/WSL）
│   └── run.ps1                 # PowerShell 脚本（Windows）
└── results/                    # 训练结果输出目录
    └── <exp_name>/
        ├── train_log.json      # 训练日志（JSON格式）
        ├── curves.png          # 训练曲线图（4个子图）
        ├── model_best.pt       # 最优模型权重
        └── model_last.pt       # 最新模型权重
```

---

## 🖥️ 硬件要求

### **最低要求**
- **CPU**：支持 Python 3.8+ 的现代 CPU
- **内存**：至少 4GB RAM
- **磁盘空间**：至少 2GB（用于数据集、模型和结果）

### **推荐配置**
- **CPU**：多核 CPU（推荐 4 核以上）
- **GPU**（可选但强烈推荐）：
  - NVIDIA GPU with CUDA 支持
  - 至少 4GB 显存（推荐 8GB+）
  - CUDA 11.0 或更高版本
- **内存**：8GB+ RAM
- **磁盘空间**：5GB+（用于完整训练过程和结果）

### **设备自动检测**
代码会自动检测可用设备：
- 优先使用 **GPU**（如果可用）
- 否则使用 **CPU**
- 支持 CUDA、MPS（Apple Silicon）和 CPU

### **性能参考**
- **CPU 训练**：约 0.5-2 step/s（取决于 CPU 性能）
- **GPU 训练**：约 10-50+ step/s（取决于 GPU 型号）
- **单次实验时长**（5000 steps）：
  - CPU：约 40-200 分钟
  - GPU：约 2-10 分钟

---

## 🔧 环境安装

### **1. Python 环境**
确保已安装 Python 3.8 或更高版本：
```bash
python --version  # 应显示 3.8+
```

### **2. 创建虚拟环境（推荐）**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS/WSL
python -m venv .venv
source .venv/bin/activate
```

### **3. 安装依赖**
```bash
pip install -r requirements.txt
```

### **4. 验证安装**
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

---

## 🚀 快速开始

### **方式一：使用一键运行脚本**

#### **Windows PowerShell**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

#### **Linux/macOS/WSL Bash**
```bash
bash scripts/run.sh
```

### **方式二：直接运行 Python 命令**
```bash
python -m src.train
```

---

## 📝 可复现实验的 Exact 命令（含随机种子）

### **⚠️ 重要：以下命令包含固定随机种子，可完全复现实验结果**

### **1. 基础语言模型（推荐配置）**

```bash
python -m src.train \
  --dataset tiny_shakespeare \
  --vocab byte \
  --seq_len 256 \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 4 \
  --ffn_hidden 1024 \
  --dropout 0.1 \
  --model_type lm \
  --attention_type standard \
  --pos_encoding_type absolute \
  --lr 3e-4 \
  --optimizer adamw \
  --weight_decay 0.01 \
  --scheduler cosine \
  --warmup_steps 500 \
  --batch_size 64 \
  --max_steps 5000 \
  --eval_interval 200 \
  --seed 3407 \
  --results_dir results/exp_lm_standard_abs_seed3407
```

**预期结果**：
- 训练时间：约 5-30 分钟（取决于硬件）
- 最终验证损失：约 1.5-2.5
- 最终困惑度：约 5-15

---

### **2. 语言模型 + 相对位置编码**

```bash
python -m src.train \
  --dataset tiny_shakespeare \
  --vocab byte \
  --seq_len 256 \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 4 \
  --ffn_hidden 1024 \
  --dropout 0.1 \
  --model_type lm \
  --attention_type standard \
  --pos_encoding_type relative \
  --max_relative_position 128 \
  --lr 3e-4 \
  --optimizer adamw \
  --weight_decay 0.01 \
  --scheduler cosine \
  --warmup_steps 500 \
  --batch_size 64 \
  --max_steps 5000 \
  --eval_interval 200 \
  --seed 3407 \
  --results_dir results/exp_lm_standard_rel_seed3407
```

---

### **3. 语言模型 + 线性注意力**

```bash
python -m src.train \
  --dataset tiny_shakespeare \
  --vocab byte \
  --seq_len 256 \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 4 \
  --ffn_hidden 1024 \
  --dropout 0.1 \
  --model_type lm \
  --attention_type linear \
  --pos_encoding_type absolute \
  --lr 3e-4 \
  --optimizer adamw \
  --weight_decay 0.01 \
  --scheduler cosine \
  --warmup_steps 500 \
  --batch_size 64 \
  --max_steps 5000 \
  --eval_interval 200 \
  --seed 3407 \
  --results_dir results/exp_lm_linear_abs_seed3407
```

---

### **4. 编码器-解码器模型（Seq2Seq）**

```bash
python -m src.train \
  --dataset tiny_shakespeare \
  --vocab byte \
  --seq_len 256 \
  --d_model 256 \
  --n_heads 4 \
  --n_encoder_layers 4 \
  --n_decoder_layers 4 \
  --ffn_hidden 1024 \
  --dropout 0.1 \
  --model_type seq2seq \
  --attention_type standard \
  --pos_encoding_type absolute \
  --lr 3e-4 \
  --optimizer adamw \
  --weight_decay 0.01 \
  --scheduler cosine \
  --warmup_steps 500 \
  --batch_size 64 \
  --max_steps 5000 \
  --eval_interval 200 \
  --seed 3407 \
  --results_dir results/exp_seq2seq_standard_abs_seed3407
```

---

### **5. 完整挑战配置（相对位置编码 + 线性注意力 + 解码器）**

```bash
python -m src.train \
  --dataset tiny_shakespeare \
  --vocab byte \
  --seq_len 256 \
  --d_model 256 \
  --n_heads 4 \
  --n_encoder_layers 4 \
  --n_decoder_layers 4 \
  --ffn_hidden 1024 \
  --dropout 0.1 \
  --model_type seq2seq \
  --attention_type linear \
  --pos_encoding_type relative \
  --max_relative_position 128 \
  --lr 3e-4 \
  --optimizer adamw \
  --weight_decay 0.01 \
  --scheduler cosine \
  --warmup_steps 500 \
  --batch_size 64 \
  --max_steps 5000 \
  --eval_interval 200 \
  --seed 3407 \
  --results_dir results/exp_seq2seq_linear_rel_seed3407
```

---

## ⚙️ 命令行参数说明

### **数据集参数**
- `--dataset`: 数据集名称（`tiny_shakespeare`）
- `--vocab`: 词汇表类型（`byte` 或 `char`）
- `--seq_len`: 序列长度（默认 `256`）

### **模型架构参数**
- `--d_model`: 模型维度（默认 `256`）
- `--n_heads`: 注意力头数量（默认 `4`）
- `--n_layers`: Transformer 层数（语言模型，默认 `4`）
- `--n_encoder_layers`: 编码器层数（Seq2Seq，默认 `4`）
- `--n_decoder_layers`: 解码器层数（Seq2Seq，默认 `4`）
- `--ffn_hidden`: 前馈网络隐藏层维度（默认 `1024`）
- `--dropout`: Dropout 比率（默认 `0.1`）

### **模型类型参数（新增）**
- `--model_type`: 模型类型
  - `lm`: 语言模型（自回归）
  - `seq2seq`: 编码器-解码器
- `--attention_type`: 注意力机制类型
  - `standard`: 标准缩放点积注意力（O(n²)）
  - `linear`: 线性注意力（O(n)，适合长序列）
- `--pos_encoding_type`: 位置编码类型
  - `absolute`: 绝对位置编码（正弦/余弦）
  - `relative`: 相对位置编码（Shaw et al.）
- `--max_relative_position`: 最大相对位置距离（仅用于相对位置编码，默认 `128`）

### **训练超参数**
- `--lr`: 学习率（默认 `3e-4`）
- `--optimizer`: 优化器（`adamw` 或 `adam`，默认 `adamw`）
- `--weight_decay`: 权重衰减（默认 `0.01`）
- `--scheduler`: 学习率调度策略
  - `cosine`: 余弦退火 + 线性 warmup（推荐）
  - `onecycle`: 单周期学习率
  - `steplr`: 阶梯式衰减
  - `none`: 固定学习率
- `--warmup_steps`: Warmup 步数（默认 `500`）

### **训练设置**
- `--batch_size`: 批次大小（默认 `64`）
- `--max_steps`: 最大训练步数（默认 `5000`）
- `--eval_interval`: 评估间隔（步数，默认 `200`）
- `--seed`: **随机种子**（默认 `3407`，用于复现实验）

### **输出设置**
- `--results_dir`: 结果保存目录（默认 `results/exp_default`）
- `--device`: 指定设备（`cpu` 或 `cuda`，默认 `None` 自动检测）

---

## 📊 输出文件说明

训练完成后，在 `results/<exp_name>/` 目录下会生成：

### **1. train_log.json**
训练和验证日志，包含每次评估的：
- `step`: 训练步数
- `train_loss`: 训练损失
- `eval_loss`: 验证损失
- `eval_ppl`: 验证困惑度（Perplexity）
- `lr`: 当前学习率

### **2. curves.png**
详细的训练曲线图（150 DPI），包含 4 个子图：
- 📊 **训练/验证 Loss 曲线**：对比训练和验证损失的变化
- 📈 **验证集 Perplexity 曲线**：困惑度下降趋势
- 📉 **Learning Rate 调度曲线**：学习率变化（对数刻度）
- 🔍 **过拟合指示器**：训练损失 - 验证损失（用于判断过拟合）

### **3. model_best.pt**
验证集上表现最好的模型权重（PyTorch state_dict 格式）

### **4. model_last.pt**
最后一次保存的模型权重

---

## 🔍 单独绘制训练曲线

如果已有训练日志，可以单独生成图表：

```bash
python -m src.train --plot results/exp_default/train_log.json

# 或指定输出路径
python -m src.train --plot results/exp_default/train_log.json output.png
```

---

## 🐛 常见问题

### **1. Windows OpenMP 错误**
如果遇到以下错误：
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**解决方案**：
- ✅ **已自动修复**：代码中已在导入 torch 之前设置 `KMP_DUPLICATE_LIB_OK=TRUE`
- 如果仍有问题，可手动设置：
  ```powershell
  $env:KMP_DUPLICATE_LIB_OK="TRUE"
  python -m src.train ...
  ```

### **2. 数据集下载失败**
**可能原因**：网络连接问题，无法访问 GitHub

**解决方案**：
1. 检查网络连接
2. 手动下载数据集：
   - 下载地址：https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
   - 保存到：`.cache/tiny_shakespeare/tinyshakespeare.txt`
3. 重新运行程序

### **3. CUDA 内存不足**
**解决方案**：
- 减小批次大小：`--batch_size 32`
- 减小模型大小：`--d_model 128` 或 `--n_layers 2`
- 使用 CPU：`--device cpu`

### **4. 导入错误（ImportError）**
**解决方案**：
- 确保从项目根目录运行：`python -m src.train`
- 确保已安装所有依赖：`pip install -r requirements.txt`
- 检查 Python 路径是否正确

---

## 📖 代码结构说明

### **核心模块**

#### **src/model.py**
- `PositionalEncoding`: 绝对位置编码（正弦/余弦）
- `RelativePositionalEncoding`: 相对位置编码
- `LinearAttention`: 线性注意力（O(n) 复杂度）
- `MultiHeadSelfAttention`: 标准多头自注意力
- `MultiHeadCrossAttention`: 交叉注意力（用于解码器）
- `TransformerBlock`: 编码器块
- `DecoderBlock`: 解码器块
- `TransformerLM`: 语言模型
- `TransformerEncoderDecoder`: 编码器-解码器模型

#### **src/train.py**
- `parse_args()`: 解析命令行参数
- `build_optimizer()`: 构建优化器
- `build_scheduler()`: 构建学习率调度器
- `evaluate()`: 评估模型性能
- `plot_curves()`: 绘制训练曲线
- `main()`: 主训练循环

#### **src/data.py**
- `CharByteTokenizer`: 字符/字节级分词器
- `LMSequenceDataset`: 语言模型数据集
- `load_dataset()`: 加载数据集并创建 DataLoader

---

## 🎯 实验复现建议

### **推荐的实验配置**

1. **基础实验（快速验证）**：
   ```bash
   python -m src.train --max_steps 1000 --batch_size 32 --seed 3407
   ```

2. **完整实验（作业要求）**：
   使用上面提供的 exact 命令（包含完整参数和随机种子）

3. **对比实验（消融研究）**：
   - 运行不同的配置（标准/线性注意力、绝对/相对位置编码）
   - 比较 `train_log.json` 中的结果
   - 查看 `curves.png` 的曲线对比

### **确保可复现性**
- ✅ **固定随机种子**：所有命令都包含 `--seed 3407`
- ✅ **固定超参数**：使用命令中指定的所有参数
- ✅ **相同环境**：使用相同的 Python 和 PyTorch 版本

---

## 📚 参考资料

- Vaswani et al., "Attention Is All You Need" (2017)
- Shaw et al., "Self-Attention with Relative Position Representations" (2018)
- Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020)

---

## 📄 许可证

本项目为作业实现，仅供学习和研究使用。

---

**最后更新**：2024年
**维护者**：Transformer 作业实现

