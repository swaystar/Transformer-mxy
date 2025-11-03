# Transformer 作业实现

本仓库实现了一个从零搭建的基于 PyTorch 的小规模 Transformer 模型，支持：
- ✅ **语言模型（Language Model）**：自回归生成
- ✅ **编码器-解码器（Encoder-Decoder）**：序列到序列任务
- ✅ **多头注意力**：标准注意力 + 线性注意力（O(n) 复杂度）
- ✅ **位置编码**：绝对位置编码 + 相对位置编码
- ✅ **完整训练流程**：学习率调度、梯度裁剪、AdamW、模型保存、训练曲线可视化

## 目录结构
- `src/` 源代码
  - `data.py` 数据集/分词器
  - `model.py` 模型实现
  - `train.py` 训练与验证
  - `utils.py` 配置与工具
- `scripts/` 运行脚本（bash/PowerShell）
- `results/` 输出目录
- `requirements.txt` 依赖

## 安装
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 一键运行
- Bash:
```bash
bash scripts/run.sh
```
- PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

## 复现实验 exact 命令（固定随机种子）

### 基础语言模型（默认配置）
```bash
python -m src.train \
  --dataset tiny_shakespeare \
  --seq_len 256 \
  --vocab byte \
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
  --results_dir results/exp_tinysha_seed3407
```

### 使用相对位置编码
```bash
python -m src.train \
  --model_type lm \
  --pos_encoding_type relative \
  --max_relative_position 128 \
  # ... 其他参数相同
```

### 使用线性注意力
```bash
python -m src.train \
  --attention_type linear \
  # ... 其他参数相同
```

### 编码器-解码器模型（Seq2Seq）
```bash
python -m src.train \
  --model_type seq2seq \
  --n_encoder_layers 4 \
  --n_decoder_layers 4 \
  --attention_type standard \
  --pos_encoding_type absolute \
  # ... 其他参数相同
```

**其他运行方式**：
- 直接运行（需在项目根目录）：`python src/train.py [参数...]`
- Jupyter Notebook：代码已兼容相对/绝对导入，可以直接运行

## 输出
- `results/<exp>/train_log.json` 训练/验证日志（包含 step、loss、ppl、lr 等信息）
- `results/<exp>/curves.png` **详细的训练曲线图**（包含4个子图）：
  - 📊 训练/验证 Loss 曲线
  - 📈 验证集 Perplexity（困惑度）曲线
  - 📉 Learning Rate 调度曲线
  - 🔍 过拟合指示器（训练损失 - 验证损失）
- `results/<exp>/model_best.pt` 最优模型（验证损失最低）
- `results/<exp>/model_last.pt` 最后一次保存的模型

### 单独绘制已有日志的图表
如果已有训练日志，可以单独生成图表：
```bash
python -m src.train --plot results/exp_default/train_log.json
# 或指定输出路径
python -m src.train --plot results/exp_default/train_log.json output.png
```

## 常见问题

### Windows OpenMP 错误
如果遇到以下错误：
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**原因**：Windows 上多个库（PyTorch、NumPy、MKL）链接了不同的 OpenMP 运行时，导致冲突。

**解决方案**：
- ✅ **已自动修复**：代码中已在导入 torch 之前设置 `KMP_DUPLICATE_LIB_OK=TRUE`
- 如果仍有问题，可手动设置环境变量：
  ```powershell
  $env:KMP_DUPLICATE_LIB_OK="TRUE"
  python -m src.train ...
  ```

### 数据集下载失败
- 检查网络连接
- 手动下载数据集到 `.cache/tiny_shakespeare/tinyshakespeare.txt`
- 数据集地址：https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

