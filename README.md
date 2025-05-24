# kanji-2-kana

日本語の漢字混じり文章をカタカナに変換するための学習・推論パイプラインです。
Gemma-3 1Bモデルをベースに、SFTおよびGRPO（改良版）で学習し、高速・高精度な推論スクリプトを提供します。

---

## 📋 必要環境

- Python 3.10以上
- CUDA対応GPU (推奨: 8GB以上)
- Git

### 依存管理 (uv)

本プロジェクトでは `uv` を用いて依存を管理します。`uv.lock` にロックされているため、以下のコマンドで環境を構築できます。

```bash
# 依存をロックファイルからインストール
uv sync
```

新しい依存を追加・更新するには：
```bash
uv add <パッケージ名>
```

使い終わったら:
```bash
uv remove <パッケージ名>
```

---

## 📂 ディレクトリ構成

```
./
├── data/                      # 元データ (JSONL形式)
├── train/                     # 学習用スクリプト
│   ├── train_gemma3_1b_kanji2kana_sft.py          # SFT学習
│   └── train_gemma3_1b_kanji2kana_grpo_improved.py # 改良版GRPO学習
├── inference/                 # 推論用スクリプト
│   ├── inference_gemma3_kanji2kana.py  # 通常推論
│   └── fast_inference.py              # 超高速推論（バッチ、pipeline、vLLM対応）
├── model/                     # 学習済みモデル保存先
│   ├── gemma3_1b_kanji2kana_sft/
│   └── gemma3_1b_kanji2kana_sft_merged/
└── README.md                  # このファイル
```

---

## 🗄️ データ準備

学習用データは `data/` 以下のJSONLファイルを使用します。各行に以下のキーを持つJSONオブジェクトを含む形式です：

```json
{
  "output": "漢字混じり文章",
  "input": "正解カタカナ文章"
}
```

ストリーミング処理でサンプリング・フィルタリングをかけながらデータセットを生成します。

- **SFT**: `train/train_gemma3_1b_kanji2kana_sft.py` 内の `load_streaming_dataset` 関数
- **GRPO**: `train/train_gemma3_1b_kanji2kana_grpo_improved.py` 内の `load_grpo_dataset` 関数

サンプリング率や最大サンプル数はスクリプト内で設定可能です。

---

## 🚀 学習

### 1. SFT (教師ありファインチューニング)

```bash
uv run python train/train_gemma3_1b_kanji2kana_sft.py
```

主な設定：
- バッチサイズ: 64
- 勾配蓄積: 2
- LoRAランク: 32
- 学習率: 8e-4

### 2. 改良版 GRPO (強化学習)

```bash
uv run python train/train_gemma3_1b_kanji2kana_grpo_improved.py
```

改良版報酬関数:
1. カタカナ純度 (句読点・記号を除外)
2. 不適切文字ペナルティ (ひらがな・漢字)
3. 長さ適切性
4. 句読点保持
5. 完全性
6. フォーマット遵守
7. 文字種一貫性
8. 特殊文字処理

設定例:
- バッチサイズ: 16
- 勾配蓄積: 4
- LoRAランク: 64
- 学習率: 5e-5

---

## 🤖 推論

### 1. 通常推論

```bash
uv run python inference/inference_gemma3_kanji2kana.py
```

- **対話モード**: 入力を逐次変換
- **バッチモード**: 複数行を一括変換
- **ファイルモード**: `inference.py input.txt output.txt`

### 2. 超高速推論

```bash
uv run python inference/fast_inference.py
```

- PyTorch 2.0 `torch.compile` でモデルコンパイル
- `max_new_tokens=50`, `temperature=0.0`, `do_sample=False`
- シーケンス長を256に短縮
- 4bit量子化 + キャッシュ最適化
- 対話 / バッチ / ベンチマークモード

### 3. Transformers Pipeline モード

```bash
uv run python inference/fast_inference.py --hf
```

- 8bit量子化＋device_map="auto"
- `text-generation` pipeline で簡易推論

### 4. vLLM モード

```bash
uv run python inference/fast_inference.py --vllm
```

- vLLM API でバッチスケジューリング
- `LLM.generate` による高速推論

### 5. 速度ベンチマーク

```bash
uv run python inference/fast_inference.py --benchmark
```

バッチ／逐次それぞれの平均推論時間と文字/秒を出力します。

---

## 🎓 使用例

```bash
# SFT学習
uv run python train/train_gemma3_1b_kanji2kana_sft.py

# 通常推論
uv run python inference/inference_gemma3_kanji2kana.py

# 超高速推論
uv run python inference/fast_inference.py

# HF pipeline推論
uv run python inference/fast_inference.py --hf

# vLLM推論
uv run python inference/fast_inference.py --vllm

# 速度ベンチマーク
uv run python inference/fast_inference.py --benchmark
```

---

## 📜 ライセンス

MIT License
