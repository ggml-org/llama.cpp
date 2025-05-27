# KV Cache Monitor

这个工具用于监控llama.cpp中的KV cache张量，支持按层过滤。

## 编译

```bash
cmake --build build-arm64 --config Release -j12
```

## 使用方法

### 监控所有层的KV cache（默认行为）
```bash
./build-arm64/bin/kv-cache-monitor -m /path/to/model.gguf -p "Hello, world"
```

### 监控特定层的KV cache
```bash
# 只监控第0层
./build-arm64/bin/kv-cache-monitor -m /path/to/model.gguf -p "Hello, world" --layer 0

# 只监控第5层
./build-arm64/bin/kv-cache-monitor -m /path/to/model.gguf -p "Hello, world" --layer 5
```

## 参数说明

- `--layer <n>`: 指定要监控的层号（从0开始）。如果不指定或设为-1，则监控所有层。

## 输出说明

工具会输出：
1. 每个KV cache张量的详细信息，包括层号、形状、数据类型
2. 统计信息：均值、标准差、最小值、最大值
3. 张量的详细数值（对于非量化类型）
4. 最终的监控摘要

## 示例输出

```
Monitoring KV cache for layer 0 only
[KV-CACHE] Layer 0 - blk.0.attn_k.weight: shape=[4096,4096,1,1] type=f16 elements=16777216
[KV-CACHE]   stats: mean=0.000123, std=0.045678, min=-0.234567, max=0.345678
...
=== KV Cache Monitoring Summary ===
Monitored layer: 0
Total callback steps: 42
KV Cache tensors encountered:
  blk.0.attn_k.weight (layer 0): 1 times
  blk.0.attn_v.weight (layer 0): 1 times
=====================================
```

这样您就可以专注于特定层的KV cache行为，而不会被其他层的输出干扰。
