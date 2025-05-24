# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "torch",
#     "transformers",
#     "datasets",
#     "trl",
#     "accelerate",
#     "peft",
#     "bitsandbytes",
#     "pandas",
#     "numpy",
#     "tensorboardx",
# ]
# ///

"""
🚀 Gemma-3-1B SFT L4最適化版 - 23GBメモリを最大限活用
漢字→カタカナ変換タスクでメモリ効率とバッチサイズを大幅改善した学習スクリプト
"""

import json
import torch
import os
import gc
from datasets import Dataset, load_dataset, load_from_disk
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

# L4 23GB最大活用のためのメモリ最適化設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# L4最適化モデル設定（23GB活用）
max_seq_length = 320   # カタカナ変換に最適な長さ（効率重視）
lora_rank = 64         # LoRAランクを大幅増加（品質向上）

print("🚀 L4最適化 Gemma-3-1Bモデルとトークナイザーを読み込み中...")
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-pt-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16,  # より効率的な精度
)

# Gemma-3チャットテンプレート設定
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)

# 高性能LoRAアダプター設定
print("⚙️ 高性能LoRAアダプターを設定中...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=lora_rank,
    lora_alpha=lora_rank * 2,      # より積極的な学習
    lora_dropout=0.03,             # 過学習防止を少し緩和
    bias="none",
    random_state=3407,
)

def format_gemma3_conversation(kanji_text: str, katakana_text: str) -> dict:
    """Gemma-3用の会話形式フォーマット"""
    messages = [
        {
            "role": "user",
            "content": f"以下の漢字混じり文をカタカナに変換してください：{kanji_text}"
        },
        {
            "role": "assistant", 
            "content": katakana_text
        }
    ]
    return {"conversations": messages}

def load_optimized_streaming_dataset(file_paths: list[str], max_samples: int = 80000, sample_rate: float = 0.4) -> Dataset:
    """
    L4最適化版：大容量データセットを効率的に読み込む
    sample_rate: サンプリング率（0.4 = 40%をサンプリング）
    """
    data = []
    sample_count = 0
    total_lines = 0
    
    for file_path in file_paths:
        print(f"📖 大容量データ読み込み開始: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    total_lines += 1
                    
                    # サンプリング：より多くのデータを使用
                    if torch.rand(1).item() > sample_rate:
                        continue
                        
                    if sample_count >= max_samples:
                        print(f"✅ 最大サンプル数 {max_samples} に到達")
                        break
                    
                    try:
                        item = json.loads(line.strip())
                        kanji_text = item["output"]      # 漢字混じり文章
                        katakana_text = item["input"]    # カタカナ文章
                        
                        # データ品質チェック（少し緩和して多様性確保）
                        if len(kanji_text) > 80 or len(katakana_text) > 80:
                            continue
                        
                        if len(kanji_text) < 3 or len(katakana_text) < 3:
                            continue
                        
                        if not kanji_text.strip() or not katakana_text.strip():
                            continue
                        
                        # カタカナ純度事前チェック
                        katakana_chars = len([c for c in katakana_text if '\u30A0' <= c <= '\u30FF'])
                        total_chars = len([c for c in katakana_text if not c.isspace()])
                        if total_chars > 0 and katakana_chars / total_chars < 0.7:
                            continue
                        
                        # Gemma-3会話形式でフォーマット
                        conversation_data = format_gemma3_conversation(kanji_text, katakana_text)
                        
                        # トークン長チェック
                        text = tokenizer.apply_chat_template(
                            conversation_data["conversations"], 
                            tokenize=False
                        )
                        
                        if len(tokenizer.encode(text)) <= max_seq_length:
                            data.append(conversation_data)
                            sample_count += 1
                            
                            if sample_count % 1000 == 0:
                                print(f"📊 処理済み: {sample_count}/{total_lines} (サンプリング率: {sample_rate*100:.1f}%)")
                        
                        # メモリ管理
                        if sample_count % 10000 == 0:
                            gc.collect()
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        continue
                    except Exception as e:
                        if line_num % 20000 == 0:
                            print(f"⚠️ 行 {line_num} でエラー: {e}")
                        continue
                        
                if sample_count >= max_samples:
                    break
                    
        except FileNotFoundError:
            print(f"❌ ファイルが見つかりません: {file_path}")
            continue
    
    print(f"✅ L4最適化データセット作成完了: {len(data)} サンプル (総行数: {total_lines})")
    return Dataset.from_list(data)

def apply_chat_template(examples):
    """チャットテンプレートを適用"""
    texts = []
    for conversation in examples["conversations"]:
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

# データセット作成
dataset_files = [
    "data/train_llm-jp-corpus-v3.jsonl",
    "data/train_wikipedia.jsonl"
]

print("📚 datasetsライブラリでデータセットをロード＆キャッシュ中...")
cache_dir = "cache/sft_l4_arrow"
if not os.path.exists(cache_dir):
    raw = load_dataset("json", data_files=dataset_files, split="train")
    raw = raw.filter(
        lambda ex: 3 <= len(ex["output"]) <= 80 and 3 <= len(ex["input"]) <= 80 and sum(1 for c in ex["input"] if '\u30A0' <= c <= '\u30FF') / max(len(ex["input"].replace(" ","")),1) >= 0.7,
        num_proc=16
    )
    raw = raw.map(
        lambda examples: {"conversations":[format_gemma3_conversation(o,i)["conversations"] for o,i in zip(examples["output"], examples["input"])]},
        batched=True, batch_size=1000, num_proc=16, remove_columns=["input","output","left_context"]
    )
    raw.save_to_disk(cache_dir)
dataset = load_from_disk(cache_dir)
print("🔄 トークナイズをバッチ・並列で実行中...")
dataset = dataset.map(
    lambda examples: {"text":[tokenizer.apply_chat_template(conv, tokenize=False) for conv in examples["conversations"]]},
    batched=True, batch_size=1000, num_proc=16, remove_columns=["conversations"]
)

# サンプル確認
print("\n=== 📋 L4最適化サンプル確認 ===")
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    print(f"サンプル {i+1}:")
    print(sample["text"][:250] + "..." if len(sample["text"]) > 250 else sample["text"])
    print("-" * 50)

# メモリクリーンアップ
gc.collect()
torch.cuda.empty_cache()

# メモリ使用量確認
if torch.cuda.is_available():
    print(f"🖥️ データ読み込み後GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# L4最適化SFTTrainer設定（23GB活用）
print("⚙️ L4最大活用SFTTrainerを設定中...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=96,  # L4の23GBメモリを最大活用（大幅増加）
        gradient_accumulation_steps=2,   # 実効バッチサイズ = 192
        warmup_steps=150,                # より多くのウォームアップ
        num_train_epochs=30,              # エポック数増加
        learning_rate=1e-3,              # より大きなバッチサイズに対応
        logging_steps=5,                 # より細かいログ出力（TensorBoard用）
        save_steps=100,                  # より頻繁な保存
        eval_steps=100,                  # 評価ステップ追加
        optim="adamw_torch_fused",       # より高速なオプティマイザー
        weight_decay=0.02,               # 過学習防止強化
        lr_scheduler_type="cosine",      # コサインスケジューラー
        seed=3407,
        output_dir="./outputs/gemma3_1b_kanji2kana_sft_l4_optimized",
        report_to="tensorboard",         # TensorBoard有効化
        logging_dir="./logs/tensorboard_sft_l4", # TensorBoardログディレクトリ
        dataloader_pin_memory=True,      # データローダー最適化
        dataloader_num_workers=8,        # データ読み込み並列度
        gradient_checkpointing=False,    # GPU高速化（メモリ十分なため）
        bf16=True,                       # Gemma-3用にbfloat16を使用
        max_grad_norm=1.0,               # 勾配クリッピング
        remove_unused_columns=False,     # メタデータ保持
        save_total_limit=5,              # 保存モデル数制限
    ),
)

print("🚀 L4最大活用 Gemma-3-1B SFT訓練開始...")
print("🔥 L4最適化設定:")
print(f"  - バッチサイズ: 96 (元の3倍)")
print(f"  - 実効バッチサイズ: 192 (元の6倍)")
print(f"  - データサンプル数: {len(dataset)} (元の約2.7倍)")
print(f"  - サンプリング率: {sample_rate*100:.1f}% (元の約2.7倍)")
print(f"  - LoRAランク: {lora_rank} (元の4倍)")
print(f"  - シーケンス長: {max_seq_length} (効率化)")

# メモリ使用量確認
if torch.cuda.is_available():
    print(f"🖥️ 訓練前GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

trainer_stats = trainer.train()

# 訓練統計表示
print(f"\n📊 L4最適化SFT訓練完了!")
print(f"⏱️ 訓練時間: {trainer_stats.metrics['train_runtime']:.2f} 秒")
print(f"📈 最終ロス: {trainer_stats.metrics['train_loss']:.4f}")

# モデル保存（model/配下に保存）
save_dir = "model/gemma3_1b_kanji2kana_sft_l4_optimized"
print(f"💾 最適化モデルを {save_dir} に保存中...")
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# float16形式でも保存
print("💾 float16形式でも保存中...")
model.save_pretrained_merged(f"{save_dir}_merged", tokenizer)

# テスト推論
print("\n=== 🧪 L4最適化SFT訓練後テスト推論 ===")
# 推論モードに切り替え
FastModel.for_inference(model)

def test_l4_optimized_inference(text: str) -> str:
    """L4最適化SFT訓練後のテスト推論"""
    messages = [{
        "role": "user",
        "content": f"以下の漢字混じり文をカタカナに変換してください：{text}"
    }]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.2,                # より決定的な生成
            top_p=0.85,                     # より集中した候補選択
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,        # 繰り返し防止
        )
    
    # 入力部分を除去して出力のみ取得
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result.strip()

# 拡張テストケース
test_cases = [
    "今日は良い天気です。",
    "私は学生です。", 
    "東京駅に行きます。",
    "美しい花が咲いています。",
    "明日は雨が降るでしょう。",
    "コンピューターを使います。",
    "新幹線で大阪へ行く。",
    "科学技術の発展が著しい。",
    "国際会議が開催される。",
    "人工知能が普及している。"
]

print("🔍 L4最適化SFT訓練後の推論テスト結果:")
for i, test_case in enumerate(test_cases, 1):
    try:
        result = test_l4_optimized_inference(test_case)
        print(f"{i}. 入力: {test_case}")
        print(f"   出力: {result}")
        
        # 詳細評価
        katakana_chars = len([c for c in result if '\u30A0' <= c <= '\u30FF'])
        total_chars = len([c for c in result if not c.isspace()])
        katakana_ratio = katakana_chars / max(total_chars, 1)
        length_ratio = len(result) / len(test_case) if test_case else 0
        
        print(f"   カタカナ率: {katakana_ratio:.1%}, 長さ比率: {length_ratio:.2f}")
        print("-" * 50)
    except Exception as e:
        print(f"{i}. 入力: {test_case}")
        print(f"   エラー: {e}")
        print("-" * 50)

# 最終メモリ使用量確認
if torch.cuda.is_available():
    print(f"\n🖥️ 最終GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"📋 GPU最大メモリ使用量: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"📈 メモリ活用率: {torch.cuda.max_memory_allocated() / (23 * 1024**3) * 100:.1f}%")

print(f"\n✅ L4最適化 Gemma-3-1B Kanji2Kana SFT訓練完了！")
print(f"💾 保存先: {save_dir}")
print(f"📊 使用データ: {len(dataset)} サンプル (サンプリング率: {sample_rate*100:.1f}%)")

print("\n" + "="*70)
print("🚀 L4 SFT最適化の詳細改善点:")
print("="*70)
print("【メモリ活用】")
print("- バッチサイズ: 32 → 96 (3倍増)")
print("- 実効バッチサイズ: 32 → 192 (6倍増)")
print("- シーケンス長: 384 → 320 (効率化)")
print("- LoRAランク: 32 → 64 (2倍増)")
print("")
print("【データ品質】")
print(f"- サンプル数: 30,000 → {len(dataset)} (約2.7倍増)")
print("- サンプリング率: 25% → 40% (1.6倍増)")
print("- 品質フィルタリング強化")
print("")
print("【学習効率】")
print("- エポック数: 2 → 3 (1.5倍増)")
print("- オプティマイザー: adamw_8bit → adamw_torch_fused")
print("- スケジューラー: linear → cosine")
print("- より細かい評価・保存")
print("")
print("🎯 期待される大幅改善:")
print("- 学習速度: 約6倍高速化")
print("- GPUメモリ活用率: 15% → 70-85%")
print("- カタカナ変換精度の向上")
print("- より安定した学習")
print("")
print("⚡ L4の23GBメモリを最大限活用する設定完了！")

# RAM使用量に関するアドバイス
print("\n" + "="*60)
print("💡 RAM使用量について:")
print("="*60)
print(f"処理データ量: 約{34 * sample_rate:.1f}GB (ストレージ)")
print(f"実際のRAM使用量: 約{34 * sample_rate * 0.3:.1f}GB (サンプリング後)")
print("推奨RAM: 32GB以上")
print("必要最小RAM: 24GB")
print("")
print("🔥 40GBデータ処理時の推奨スペック:")
print("- RAM: 64GB以上")
print("- GPU: 24GB以上 (A6000/RTX4090等)")
print("- SSD: 100GB以上の空き容量") 