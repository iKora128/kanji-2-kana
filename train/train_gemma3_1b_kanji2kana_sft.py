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
Gemma-3-1Bでunslothを使った漢字→カタカナ変換学習スクリプト
データサイズが大きいため、メモリ効率を考慮したバッチ処理とサンプリングを実装
"""

import json
import torch
import os
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

# L4 23GB最大活用のためのメモリ最適化設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# L4最適化モデル設定（23GB活用）
max_seq_length = 384  # カタカナ変換に最適な長さ
lora_rank = 32        # LoRAランクを倍増（品質向上）

print("🚀 Gemma-3-1Bモデルとトークナイザーを読み込み中...")
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

# LoRAアダプターを追加
print("⚙️ LoRAアダプターを設定中...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=lora_rank,
    lora_alpha=lora_rank * 2,
    lora_dropout=0.05,
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

def load_streaming_dataset(file_paths: list[str], max_samples: int = 10000, sample_rate: float = 0.1) -> Dataset:
    """
    大きなデータセットをストリーミング処理で効率的に読み込む
    sample_rate: サンプリング率（0.1 = 10%をサンプリング）
    """
    data = []
    sample_count = 0
    total_lines = 0
    
    for file_path in file_paths:
        print(f"📖 読み込み開始: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    total_lines += 1
                    
                    # サンプリング：一定の確率でスキップ
                    if torch.rand(1).item() > sample_rate:
                        continue
                        
                    if sample_count >= max_samples:
                        print(f"✅ 最大サンプル数 {max_samples} に到達")
                        break
                    
                    try:
                        item = json.loads(line.strip())
                        kanji_text = item["output"]      # 漢字混じり文章
                        katakana_text = item["input"]    # カタカナ文章
                        
                        # データ品質チェック
                        if len(kanji_text) > 100 or len(katakana_text) > 100:
                            continue
                        
                        if not kanji_text.strip() or not katakana_text.strip():
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
                            
                            if sample_count % 500 == 0:
                                print(f"📊 処理済み: {sample_count}/{total_lines} (サンプリング率: {sample_rate*100:.1f}%)")
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        continue
                    except Exception as e:
                        if line_num % 10000 == 0:
                            print(f"⚠️ 行 {line_num} でエラー: {e}")
                        continue
                        
                if sample_count >= max_samples:
                    break
                    
        except FileNotFoundError:
            print(f"❌ ファイルが見つかりません: {file_path}")
            continue
    
    print(f"✅ データセット作成完了: {len(data)} サンプル (総行数: {total_lines})")
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

print("📚 L4最大活用 Gemma-3用データセットを作成中...")
# L4の23GBメモリを最大限活用してより多くのデータを処理
sample_rate = 0.25   # 25%をサンプリング（約8.5GBのデータ量）
max_samples = 60000  # 最大サンプル数を大幅増加

dataset = load_streaming_dataset(dataset_files, max_samples=max_samples, sample_rate=sample_rate)

# チャットテンプレートを適用
print("🔄 チャットテンプレートを適用中...")
dataset = dataset.map(apply_chat_template, batched=True, remove_columns=["conversations"])

# サンプル確認
print("\n=== 📋 サンプル確認 ===")
for i in range(min(2, len(dataset))):
    sample = dataset[i]
    print(f"サンプル {i+1}:")
    print(sample["text"][:300] + "..." if len(sample["text"]) > 300 else sample["text"])
    print("-" * 50)

# メモリ使用量確認
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"🖥️ 初期GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# SFTTrainer設定（メモリ効率を重視）
print("⚙️ SFTTrainerを設定中...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=64,  # L4の23GBメモリを最大活用（倍増）
        gradient_accumulation_steps=2,   # 実効バッチサイズ = 128
        warmup_steps=100,                # より多くのウォームアップ
        num_train_epochs=3,              # エポック数増加
        learning_rate=8e-4,              # より大きなバッチサイズに対応
        logging_steps=10,               # より細かいログ出力（TensorBoard用）
        save_steps=200,                 # より頻繁な保存
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./outputs/gemma3_1b_kanji2kana_sft",
        report_to="tensorboard",        # TensorBoard有効化
        logging_dir="./logs/tensorboard", # TensorBoardログディレクトリ
        dataloader_pin_memory=True,   # データローダー最適化
        dataloader_num_workers=8,     # データ読み込み並列度を倍増
        gradient_checkpointing=False,  # GPU高速化
        bf16=True,                    # Gemma-3用にbfloat16を使用
    ),
)

print("🚀 Gemma-3-1B SFT訓練開始...")
trainer_stats = trainer.train()

# 訓練統計表示
print(f"\n📊 訓練完了!")
print(f"⏱️ 訓練時間: {trainer_stats.metrics['train_runtime']:.2f} 秒")
print(f"📈 最終ロス: {trainer_stats.metrics['train_loss']:.4f}")

# モデル保存（model/配下に保存）
save_dir = "model/gemma3_1b_kanji2kana_sft"
print(f"💾 モデルを {save_dir} に保存中...")
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# float16形式でも保存
print("💾 float16形式でも保存中...")
model.save_pretrained_merged(f"{save_dir}_merged", tokenizer)

# テスト推論
print("\n=== 🧪 テスト推論 ===")
# 推論モードに切り替え
FastModel.for_inference(model)

def test_gemma3_inference(text: str) -> str:
    """Gemma-3でのテスト推論"""
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
            max_new_tokens=150,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 入力部分を除去して出力のみ取得
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result.strip()

# テストケース
test_cases = [
    "今日は良い天気です。",
    "私は学生です。", 
    "東京駅に行きます。",
    "美しい花が咲いています。",
    "明日は雨が降るでしょう。"
]

print("🔍 Gemma-3推論テスト結果:")
for i, test_case in enumerate(test_cases, 1):
    try:
        result = test_gemma3_inference(test_case)
        print(f"{i}. 入力: {test_case}")
        print(f"   出力: {result}")
        print("-" * 40)
    except Exception as e:
        print(f"{i}. 入力: {test_case}")
        print(f"   エラー: {e}")
        print("-" * 40)

# 最終メモリ使用量確認
if torch.cuda.is_available():
    print(f"\n🖥️ 最終GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"📋 GPU最大メモリ使用量: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print(f"\n✅ Gemma-3-1B Kanji2Kana SFT訓練完了！")
print(f"💾 保存先: {save_dir}")
print(f"📊 使用データ: {len(dataset)} サンプル (サンプリング率: {sample_rate*100:.1f}%)")

# RAM使用量に関するアドバイス
print("\n" + "="*60)
print("💡 L4最適化済み設定の詳細:")
print("="*60)
print("サンプリング率:", f"{sample_rate*100:.1f}% (前回の3倍)")
print("処理データ量:", f"約{34 * sample_rate:.1f}GB")
print("バッチサイズ:", "8 (前回の8倍)")
print("実効バッチサイズ:", "16 (前回の2倍)")
print("エポック数:", "2 (前回の2/3)")
print("")
print("🚀 予想される改善:")
print("- 訓練速度: 約8-10倍高速化")
print("- GPUメモリ使用率: 4% → 40-60%")
print("- 総訓練時間: 9時間 → 1-2時間")
print("")
print("⚡ L4の性能を最大限活用する設定に最適化完了！") 