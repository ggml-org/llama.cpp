# Gemma 4 12B 改造計画：Japanese Full-Duplex Spoken Dialogue System

> J-Moshi の知見を踏まえ、Gemma 4 12B Unified をベースに  
> Moshi 相当の full-duplex 音声対話システムを構築するための設計書

---

## 0. 前提整理：Gemma 4 12B と Moshi の本質的な差分

| 項目 | Moshi (J-Moshi) | Gemma 4 12B Unified |
|------|----------------|---------------------|
| 音声入力方式 | Mimi (SEANet + RVQ) → 離散トークン列 | 40ms フレームを線形投影 → 連続ベクトル列 |
| 音声の粒度 | 80ms / frame（12.5Hz） | **40ms / frame（25Hz）** |
| LLM バックボーン | 7B Temporal Transformer + Depth Transformer | **12B decoder-only Transformer** |
| 出力 | 音声トークン + テキストトークン（同時） | テキストトークンのみ |
| ストリーミング | ネイティブ対応 | KV キャッシュあり・causal attention あり → **対応可能** |
| 日本語能力 | J-Moshi：J-CHAT 69,000h で事前学習 | 140 言語対応（日本語高品質） |
| 公開ライセンス | Apache 2.0 | **Apache 2.0** |

### 結論

Gemma 4 12B は「音声を 40ms チャンクで逐次処理できる decoder-only LLM」であり、
アーキテクチャ的には **Moshi の Temporal Transformer と同等の役割** を果たせる素地がある。
不足しているのは以下の 2 点のみ：

1. **音声出力機能**（今回のスコープ外：テキスト出力で代替）
2. **Depth Transformer**（8 層音声トークン生成器：テキスト出力なら不要）

つまり「テキスト応答でよい」という前提のもとでは、
**Gemma 4 12B + ストリーミング推論パイプライン + fine-tuning** で
full-duplex に近い音声対話システムが構築できる。

---

## 1. ターゲットアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                  音声入力ストリーム（16kHz）                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ 40ms チャンク
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Gemma 4 12B 音声投影層（既存）                    │
│  raw audio (640 floats) → LLM hidden dim ベクトル            │
└──────────────────────────┬──────────────────────────────────┘
                           │ 連続ベクトル列（インクリメンタル）
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Gemma 4 12B Decoder（既存重みを流用）             │
│  causal attention + KV キャッシュ → ストリーミング処理         │
└──────────────────────────┬──────────────────────────────────┘
                           │ テキストトークン（ストリーミング）
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     テキスト応答出力                           │
│              （将来的に TTS と組み合わせて音声化）              │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 Moshi との役割対応表

| Moshi コンポーネント | 本計画での対応 | 備考 |
|---------------------|-------------|------|
| Mimi エンコーダ | Gemma 4 音声投影層 | 離散化しない点が異なる |
| Temporal Transformer | Gemma 4 12B Decoder | 重みをそのまま流用 |
| Depth Transformer | **省略** | テキスト出力なら不要 |
| Mimi デコーダ | **省略 → 将来 TTS で代替** | 音声出力は Phase 2 以降 |
| inner monologue | Gemma 4 のテキスト出力 | 役割は同じ |

---

## 2. 技術的課題と解決方針

### 課題 1：連続ベクトル vs 離散トークンの混在

Gemma 4 の既存の学習では、音声は「塊で入力・テキストで一括応答」として学習されている。
これを「チャンク単位で逐次入力・逐次応答」に変えるには fine-tuning が必要。

**解決方針**
- 音声投影層の出力を KV キャッシュに逐次 append する推論ループを実装
- 「音声 N チャンク来たらテキストを 1 トークン出力する」タイミング制御を学習させる
- J-Moshi と同様に **PAD トークンで音声とテキストのタイミングを同期**する

### 課題 2：バックチャネル・発話オーバーラップのモデリング

Moshi の full-duplex の核心は「ユーザーが話している間にも出力できる」点。
Gemma 4 はシングルストリームなので、2 チャンネルの同時モデリングが必要。

**解決方針（2 段階）**

**Phase 1（MVP）：半 full-duplex**
- ユーザー音声チャンクを逐次受け取り、無音区間を VAD で検出して割り込み点を判定
- VAD（Voice Activity Detection）で入力エネルギーを監視し、発話終了を即時検出
- Moshi ほどリアルタイムではないが、半二重より大幅に応答が速い

**Phase 2：真の full-duplex**
- ユーザーチャンネルとシステムチャンネルの 2 系列を単一の context に結合
- 特殊トークン `<USR>` / `<SYS>` でチャンネルを区別
- J-Moshi と同様にステレオ対話データで fine-tuning

### 課題 3：推論レイテンシ

Gemma 4 12B は Moshi の 7B より大きいため、単純に遅くなりうる。

**解決方針**
- 4bit 量子化（GPTQ / AWQ）で 16GB VRAM に収める
- KV キャッシュの sliding window を活用（Gemma 4 はハイブリッド attention 採用済み）
- 音声チャンクを 40ms ごとに投入し、テキストトークンが出たら即時返却

---

## 3. 学習計画

J-Moshi の 2 段階学習を踏襲しつつ、Gemma 4 の特性に合わせて調整する。

### Phase 0：推論パイプラインの構築（学習なし）

まず既存の Gemma 4 12B の重みを一切変えず、
**ストリーミング推論ループだけを実装**して動作確認する。

```python
# 概念的な実装イメージ
audio_chunks = stream_audio_16khz(mic)   # 40ms ごとに yield
kv_cache = init_kv_cache()

for chunk in audio_chunks:
    audio_embed = gemma4.project_audio(chunk)   # 音声投影層
    kv_cache = gemma4.append_to_kv(audio_embed, kv_cache)
    
    # VAD で発話区切りを検出したらテキスト生成
    if vad.is_speech_end(chunk):
        response_tokens = gemma4.generate(kv_cache)
        stream_output(response_tokens)
```

### Phase 1：Incremental Audio Fine-tuning（軽量）

**目的**：音声チャンクを逐次受け取りながら応答するタイミングを学習

**データ**
- J-CHAT（69,000h）の一部を 40ms チャンク列に変換
- 各発話ターンの開始・終了タイミングを正解ラベルとして付与

**学習設定（推奨）**

| 設定 | 値 |
|------|-----|
| 学習対象パラメータ | 音声投影層 + 最終 4 層（LoRA） |
| LoRA rank | 16〜32 |
| バッチサイズ | 64 samples |
| GPU | A100 40GB × 8 程度 |
| 学習率 | 1e-5（投影層） / 2e-6（Decoder） |
| 推定学習時間 | 12〜24h |

**凍結する重み**：Decoder の大部分（Gemma 4 の日本語・推論能力を維持）

### Phase 2：Stereo Dialogue Fine-tuning（J-Moshi 相当）

**目的**：2 チャンネル同時音声から full-duplex 応答を学習

**データ**（J-Moshi の fine-tuning データを流用可能）

| コーパス | 時間 |
|---------|------|
| Japanese Callhome | 16h |
| CSJ（2 話者部分） | 12h |
| Travel Agency Dialogue | 115h |
| Casual Dialogue（社内） | 148h |
| Consultation Dialogue（社内） | 53h |
| **合計** | **344h** |

**チャンネル表現方式**
```
<USR> [40ms audio embed] [40ms audio embed] ... </USR>
<SYS> [text token] [text token] ... </SYS>
```
→ 時間軸に沿って USR/SYS トークンを交互に並べ、
  causal attention でシステムが「ユーザーの今」を見ながら応答を生成

### Phase 3：Synthetic Data Augmentation（任意）

J-Moshi の multi-stream TTS に相当する合成データ拡張。

- Gemma 4 自身を使って口語テキストを生成（J-Moshi では Gemma-2-27b を使用）
- Voicevox / COEIROINK などの OSS TTS でステレオ音声を合成
- WER フィルタリングで品質管理（J-Moshi 実績：WER 24.6%）

---

## 4. 評価方法

J-Moshi の評価手法をそのまま採用する。

### 自動評価

- **PPL（Perplexity）**：ASR 結果に対する言語モデルスコア
  - ベースライン：dGSLM（J-Moshi 論文の比較対象）
  - 目標：J-Moshi（τ=0.8）程度の PPL を下回る

### 人手評価（クラウドソーシング）

J-Moshi 論文と同じ 5 段階評価：

| 指標 | 定義 |
|------|------|
| Naturalness | 対話がどれだけ自然に聞こえるか |
| Meaningfulness | 発話内容がどれだけ理解できるか |

**目標スコア**（J-Moshi 実績を超える）

| モデル | Naturalness | Meaningfulness |
|--------|-------------|----------------|
| dGSLM（ベースライン） | 2.44 | 1.76 |
| J-Moshi | 2.67 | 2.19 |
| **本計画目標** | **≥ 2.80** | **≥ 2.50** |

### Turn-taking 分析

J-Moshi 論文 Table 3 と同じ指標で比較：

- IPU（Inter-Pausal Unit）：発話セグメント長
- Pause：同話者間の無音
- Gap：話者交代時の無音
- **Overlap**：発話オーバーラップ時間（full-duplex の主要指標）

---

## 5. 実装ロードマップ

```
Month 1：Phase 0（推論パイプライン）
  ├─ Gemma 4 12B のストリーミング推論ループ実装
  ├─ VAD 統合（Silero-VAD 推奨）
  └─ 動作確認・レイテンシ計測

Month 2：Phase 1（軽量 fine-tuning）
  ├─ J-CHAT の 40ms チャンク変換パイプライン構築
  ├─ LoRA fine-tuning（A100 × 8、推定 12〜24h）
  └─ Phase 0 との比較評価

Month 3：Phase 2（ステレオ対話 fine-tuning）
  ├─ 344h ステレオデータの前処理
  ├─ 2 チャンネル入力のトークン化
  └─ Full-duplex fine-tuning + 評価

Month 4〜（任意）：Phase 3 + 音声出力
  ├─ 合成データ拡張
  └─ TTS 統合による音声出力化
```

---

## 6. 既存 Moshi 実装との相違点まとめ

| 項目 | J-Moshi | 本計画 |
|------|---------|--------|
| 音声入力 | 離散トークン（RVQ） | 連続ベクトル（線形投影） |
| 音声出力 | あり（Mimi デコーダ） | **なし（テキストのみ、TTS で代替）** |
| モデルサイズ | 7B | **12B** |
| 日本語事前学習 | J-CHAT 69,000h から | **Gemma 4 の事前学習を流用** |
| 最小 fine-tuning データ | 344h ステレオ | **LoRA なら 344h で十分と予想** |
| 必要 GPU（学習） | V100 128 枚 | **A100 8〜16 枚（LoRA 前提）** |
| 必要 VRAM（推論） | 非公開 | **16GB（4bit 量子化時）** |

---

## 7. リスクと留意事項

### 技術リスク

1. **連続ベクトルのインクリメンタル処理**：Gemma 4 の音声投影層は「塊入力」を想定して学習されているため、40ms チャンク単位での逐次投入でどれだけ精度が落ちるか事前評価が必要
2. **レイテンシ**：12B モデルの自己回帰生成は 7B より遅い。量子化と KV キャッシュ管理の最適化が必須
3. **バックチャネル**：真の full-duplex（発話オーバーラップ）は Phase 2 以降。Phase 1 は VAD 頼みになる

### ライセンス

- Gemma 4 12B：Apache 2.0（商用利用可）
- J-CHAT：YouTube/Podcast 由来のため、学術目的以外の利用は要確認
- ステレオ対話コーパス：各コーパスのライセンスを個別に確認

---

## 8. 参考文献

- Ohashi et al., "Towards a Japanese Full-duplex Spoken Dialogue System" (arXiv:2506.02979, 2025)
- Défossez et al., "Moshi: a speech-text foundation model for real-time dialogue" (arXiv:2410.00037, 2024)
- Google DeepMind, "Gemma 4 12B Developer Guide" (developers.googleblog.com, 2026)
- Nakata et al., "J-CHAT: Japanese Large-scale Spoken Dialogue Corpus" (arXiv:2407.15828, 2024)
