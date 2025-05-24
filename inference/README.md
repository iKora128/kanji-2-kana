# 🚀 Gemma-3-1B 漢字→カタカナ変換推論

学習済みのGemma-3-1Bモデルを使用した漢字からカタカナへの変換推論システムです。

## 📋 必要な環境

- Python 3.10以上
- CUDA対応GPU (推奨: 4GB以上)
- 学習済みモデル: `model/gemma3_1b_kanji2kana_sft/`

## 🚀 クイックスタート

### 1. 対話モード（推奨）

```bash
uv run inference/inference_gemma3_kanji2kana.py
```

対話的に文章を入力してカタカナ変換を行います。

```
📝 変換したい文章を入力: 今日は良い天気です。
🔄 変換中...
📤 入力: 今日は良い天気です。
📥 出力: キョウハヨイテンキデス。
📊 カタカナ率: 100.0%
```

### 2. シンプル推論

```bash
uv run inference/simple_inference.py
```

プリセットされたテスト文章で簡単に動作確認できます。

### 3. ファイル処理

```bash
uv run inference/inference_gemma3_kanji2kana.py input.txt output.txt
```

ファイルの全行を一括変換します。

## 📖 詳細な使い方

### 対話モード

```bash
uv run inference/inference_gemma3_kanji2kana.py
```

**利用可能なコマンド:**
- 文章入力: そのまま漢字混じり文を入力
- `batch`: バッチ処理モードに切り替え
- `quit`, `exit`, `q`: 終了

**バッチ処理モード:**
複数の文章を一度に処理できます。

```
📦 バッチ処理モード
📝 変換したい文章を入力してください (空行で処理開始):
今日は良い天気です。
私は学生です。
東京駅に行きます。
[空行を入力]
```

### ファイル処理モード

**入力ファイル例 (input.txt):**
```
今日は良い天気です。
私は学生です。
東京駅に行きます。
```

**実行:**
```bash
uv run inference/inference_gemma3_kanji2kana.py input.txt output.txt
```

**出力ファイル例 (output.txt):**
```
入力: 今日は良い天気です。
出力: キョウハヨイテンキデス。
----------------------------------------
入力: 私は学生です。
出力: ワタシハガクセイデス。
----------------------------------------
```

### カスタムモデル使用

```bash
uv run inference/inference_gemma3_kanji2kana.py --model path/to/your/model
```

## 🔧 プログラム内での使用

### シンプルな関数として使用

```python
from inference.simple_inference import convert_to_katakana

# 単発変換
result = convert_to_katakana("今日は良い天気です。")
print(result)  # キョウハヨイテンキデス。

# カスタムモデルパス
result = convert_to_katakana("こんにちは", model_path="path/to/model")
```

### 高度な推論設定

```python
from inference.inference_gemma3_kanji2kana import load_trained_model, kanji_to_katakana

# モデル一度だけ読み込み（効率的）
model, tokenizer = load_trained_model("model/gemma3_1b_kanji2kana_sft")

# 複数回推論
texts = ["今日は良い天気です。", "私は学生です。"]
for text in texts:
    result = kanji_to_katakana(model, tokenizer, text)
    print(f"{text} → {result}")
```

## 📊 性能と品質

### GPU使用量
- メモリ使用量: 約2-3GB (4bit量子化)
- 推論速度: 1文あたり約1-2秒

### 変換品質
- カタカナ純度: 通常95%以上
- 長さ適切性: 原文の80-200%
- 日本語文章に最適化

### テスト結果例

| 入力 | 出力 | カタカナ率 |
|------|------|-----------|
| 今日は良い天気です。 | キョウハヨイテンキデス。 | 100% |
| 私は学生です。 | ワタシハガクセイデス。 | 100% |
| 東京駅に行きます。 | トウキョウエキニユキマス。 | 100% |

## 🛠️ トラブルシューティング

### よくあるエラー

**1. モデルが見つからない**
```
❌ モデル読み込みエラー: model/gemma3_1b_kanji2kana_sft が見つかりません
```
→ 学習済みモデルの保存先を確認してください

**2. GPU メモリ不足**
```
CUDA out of memory
```
→ 他のプロセスを終了するか、バッチサイズを下げてください

**3. 依存関係エラー**
```
ModuleNotFoundError: No module named 'unsloth'
```
→ 以下で依存関係をインストール:
```bash
uv add unsloth torch transformers
```

### デバッグ情報

```bash
# GPU状態確認
nvidia-smi

# Python環境確認
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 🎯 使用例・応用

### 1. 読み上げ準備
音声合成の前処理として漢字をカタカナに変換

### 2. ふりがな生成
教育コンテンツでのルビ生成

### 3. テキスト正規化
データ処理パイプラインでの文字種統一

### 4. アクセシビリティ
視覚障害者向けのテキスト読み上げ準備

## 🚀 パフォーマンス最適化

### バッチ処理で高速化
```python
# 一度にモデル読み込み
model, tokenizer = load_trained_model()

# 複数文章を効率的に処理
texts = ["文章1", "文章2", "文章3"]
results = []
for text in texts:
    result = kanji_to_katakana(model, tokenizer, text)
    results.append(result)
```

### メモリ効率化
```python
import torch

# 推論後にメモリクリア
torch.cuda.empty_cache()
```

## 📝 ライセンス

このプロジェクトは学習・研究目的で作成されています。商用利用前にGemma-3モデルのライセンスを確認してください。 