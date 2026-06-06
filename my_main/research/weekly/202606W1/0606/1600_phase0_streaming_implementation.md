# Phase 0 実装ログ: Gemma 4 UA インクリメンタル音声ストリーミング

**日付**: 2026-06-06  
**担当**: Anderson  
**スクリプト**: `my_main/scripts/phase0_streaming.py`

---

## 目的

Gemma 4 12B Unified Audio を使い、音声を 40ms チャンクごとに LLM の KV キャッシュへ逐次注入できるかを検証する（Phase 0: 学習なし・配管検証のみ）。

---

## 実装内容

### アーキテクチャ

```
WAV (16kHz) -> 40ms チャンク (640 samples)
    -> rms_norm + mm.a.input_projection.weight (640 -> 3840)
    -> llama_decode(embd) x N フレーム  <- KV キャッシュへ逐次 append
    -> VAD で発話終端を検出
    -> テキストトークン生成 -> stdout
```

### 主要パラメータ (Gemma 4 UA)

| パラメータ | 値 |
|---|---|
| サンプルレート | 16,000 Hz |
| フレームサイズ | 640 samples (40ms) |
| RMS norm eps | 1e-6 |
| n_embd | 3,840 |
| 音声投影重みテンソル名 | `mm.a.input_projection.weight` |
| 量子化形式 | Q8_0 |

### 実装の要点

**GGUF パーサ (インライン実装)**
- `gguf` Python パッケージに依存せずバイナリを直接パース
- Q8_0 逆量子化をベクトル演算で実装 (block = 2B scale + 32B int8)

**音声投影 (numpy)**
```python
rms   = sqrt(mean(chunk**2) + eps)
normed = chunk / rms
embed  = proj_w @ normed   # (3840, 640) @ (640,) -> (3840,)
```

**Energy VAD**
- ファイル全体の平均エネルギーを基準値として使用
- `threshold=0.003` (基準値の 0.3%) で発話/無音を判定
- 12 フレーム (~480ms) の無音継続で発話終端を判定

**KV キャッシュクリア (llama-cpp-python 0.3.26 API)**
```python
mem = lc.llama_get_memory(ctx)
if mem is not None:
    lc.llama_memory_clear(mem, True)
```
- `llama_kv_cache_clear` は削除済み (新 API: `llama_memory_clear`)

**EOS 判定**
```python
lc.llama_vocab_is_eog(vocab, new_tok)
```
- `model.token_eos()` では Gemma 4 の `<end_of_turn>` を捕捉できず無限ループが発生
- `llama_vocab_is_eog` はモデルの全終端トークンセットを一括チェック

---

## 実行結果

### 環境
- モデル: `gemma-4-12B-it-Q4_K_M.gguf`
- mmproj: `mmproj-gemma-4-12B-it-Q8_0.gguf`
- テスト音声: `my_main/gemma4_audio_qa_input.wav` (16kHz, 10秒)
- Python: 3.11.9 (pyenv)

### ログ
```
[projector] mm.a.input_projection.weight shape=(3840, 640)  n_embd=3840
[audio] 10.00s  250 frames @ 40ms  ref_energy=0.00159
[  7560/10000ms] utterance_end
[VAD] utterance: 92 frames (3680ms)
  [inject] 92 frames (3680ms) into KV cache... 3.10s
  [logits] first 10: ['-28.71', '0.05', '-6.22', ...]  nan=False
[model] <|channel>thought
<channel|>世界で一番高い山は**エベレスト**（チョモランマ）です。
エベレストは、**ネパール**と**中国**の国境に位置しています。
```

### 確認できたこと

| 項目 | 結果 | 備考 |
|---|---|---|
| 40ms チャンク分割 | ✅ | 250 フレーム正常処理 |
| Energy VAD | ✅ | 発話区間 3680ms を正確に検出 |
| Q8_0 逆量子化 | ✅ | 投影重みを float32 に変換成功 |
| KV キャッシュへの逐次注入 | ✅ | 92 フレームを 3.10s で注入 |
| logit の有効性 | ✅ | NaN なし、正常な分布 |
| 音声理解 (ベースモデル) | ✅ | 音声内容（エベレスト）を正しく理解・回答 |
| テキスト生成の終端 | ✅ (修正後) | `llama_vocab_is_eog` で解決 |

---

## 発生したバグと修正

### 1. `llama_kv_cache_clear` が存在しない
- **原因**: llama-cpp-python 0.3.26 で API 名が変更
- **修正**: `llama_memory_clear(llama_get_memory(ctx), True)` に変更

### 2. サンプラーが assert で落ちる
- **原因**: `top_p` が NaN logit を受け取り確率分布が空になった
- **修正**: `top_k(40)` + `greedy` に変更（ファインチューニングなしでの暫定対応）

### 3. 出力が無限ループ
- **原因**: `model.token_eos()` のみでは Gemma 4 の `<end_of_turn>` を EOS と認識しない
- **修正**: `llama_vocab_is_eog(vocab, token)` に変更

---

## 性能メモ

- 92 フレーム注入: **3.10s** → 1 フレームあたり ~34ms
- フレーム 1 枚 (40ms 音声) の処理に実時間と同程度のコストがかかっている
- Phase 1 以降では量子化レベルや Flash Attention の検討が必要

---

## 次のステップ

- **Phase 1**: J-CHAT の 40ms チャンク変換パイプライン構築 + LoRA ファインチューニング
  - 音声投影層の出力に特殊トークン (`<USR>` 等) を組み合わせたデータ形式の設計
  - タイミング制御（音声 N チャンク → テキスト 1 トークン）の学習
- **改善候補 (Phase 0 のまま)**:
  - VAD を Silero-VAD に置き換えて精度向上
  - マイク入力対応 (`sounddevice`)
  - Flash Attention 有効化でレイテンシ改善

---

## 実行コマンド

```bash
/Users/andersonkaina/.pyenv/versions/3.11.9/bin/python3 \
  my_main/scripts/phase0_streaming.py \
  --model  ~/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/44ee90c4b61e888ac5b318a54ec7a94df61e9cd7/gemma-4-12B-it-Q4_K_M.gguf \
  --mmproj ~/.cache/huggingface/hub/models--ggml-org--gemma-4-12B-it-GGUF/snapshots/44ee90c4b61e888ac5b318a54ec7a94df61e9cd7/mmproj-gemma-4-12B-it-Q8_0.gguf \
  --audio  my_main/sample/gemma4_audio_qa_input.wav
```
