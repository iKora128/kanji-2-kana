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

# 定数設定
MAX_TRAIN_SAMPLES: int = 1000000   # 学習に使用するサンプル上限を100万件に変更

# L4最適化モデル設定（23GB活用）
max_seq_length = 320   # カタカナ変換に最適な長さ（効率重視）
lora_rank = 8          # LoRAランクを最適化（漢字→カタカナに十分）
lora_alpha = 8        # より適切なalpha値

print("🚀 L4最適化 Gemma-3-1Bモデルとトークナイザーを読み込み中...")
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-pt-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
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
    lora_alpha=lora_alpha,
    lora_dropout=0.01,         
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
        batched=True, batch_size=2000, num_proc=8, remove_columns=["input","output","left_context"]
    )
    raw.save_to_disk(cache_dir)
dataset = load_from_disk(cache_dir)

# 学習用サンプルを制限
dataset = dataset.select(range(min(MAX_TRAIN_SAMPLES, len(dataset))))
print(f"📊 データセットサイズ: {len(dataset)} 件")

print("🔄 トークナイズとバッチパディング／トランケーションを並列実行中...")
dataset = dataset.map(
    lambda examples: tokenizer(
        [tokenizer.apply_chat_template(conv, tokenize=False) for conv in examples["conversations"]],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    ),
    batched=True,
    batch_size=2000,
    num_proc=16,
    remove_columns=["conversations"]
)

# サンプル確認
print("\n=== L4最適化サンプル確認 ===")
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    # input_ids をデコードしてテキスト出力
    decoded = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    print(f"サンプル {i+1}:")
    print(decoded[:250] + "..." if len(decoded) > 250 else decoded)
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
        dataset_text_field="input_ids",
        per_device_train_batch_size=128,
        gradient_accumulation_steps=4,
        warmup_steps=150,
        num_train_epochs=3,
        learning_rate=5e-4,
        logging_steps=100,
        save_steps=500,
        eval_steps=1000,
        optim = "adamw_8bit",
        weight_decay=0.02,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="./outputs/gemma3_1b_kanji2kana_sft_l4_optimized",
        report_to="tensorboard",
        logging_dir="./logs/tensorboard_sft_l4",
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        gradient_checkpointing=True,  # グラデーションチェックポイントを有効化
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        save_total_limit=20,
    ),
)

print("🚀 L4最大活用 Gemma-3-1B SFT訓練開始...")
print("🔥 L4最適化設定:")
print(f"  - バッチサイズ: {trainer.args.per_device_train_batch_size}")
print(f"  - 実効バッチサイズ: {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}")
print(f"  - データサンプル数: {len(dataset)}")
print(f"  - LoRAランク: {lora_rank}")
print(f"  - シーケンス長: {max_seq_length}")

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
print(f"📊 使用データ: {len(dataset)} サンプル")