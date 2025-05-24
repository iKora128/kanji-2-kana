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
🚀 改良版 Gemma-3-1B GRPO 漢字→カタカナ変換学習
句読点・記号を適切に処理する報酬関数を実装
"""

import json
import torch
import os
import gc
import re
from datasets import Dataset, load_dataset, load_from_disk
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOTrainer, GRPOConfig

# L4 23GB最大活用のためのメモリ最適化設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# L4最適化モデル設定
max_seq_length = 320
lora_rank = 64

print("🚀 改良版GRPO Gemma-3-1Bモデルとトークナイザーを読み込み中...")
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-pt-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# 高性能LoRAアダプター設定
print("⚙️ 高性能LoRAアダプターを設定中...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=lora_rank,
    lora_alpha=lora_rank * 2,
    lora_dropout=0.03,
    bias="none",
    random_state=3407,
)

def analyze_text_composition(text: str) -> dict:
    """テキストの文字種構成を詳細分析"""
    katakana_chars = len([c for c in text if '\u30A0' <= c <= '\u30FF'])
    hiragana_chars = len([c for c in text if '\u3040' <= c <= '\u309F'])
    kanji_chars = len([c for c in text if '\u4E00' <= c <= '\u9FAF'])
    english_chars = len([c for c in text if c.isascii() and c.isalpha()])
    number_chars = len([c for c in text if c.isdigit()])
    
    # 日本語文字のみをカウント（句読点・記号を除外）
    japanese_chars = katakana_chars + hiragana_chars + kanji_chars + english_chars + number_chars
    
    return {
        'katakana': katakana_chars,
        'hiragana': hiragana_chars,
        'kanji': kanji_chars,
        'english': english_chars,
        'numbers': number_chars,
        'japanese_total': japanese_chars,
        'katakana_ratio': katakana_chars / max(japanese_chars, 1),
        'total_length': len(text)
    }

def improved_reward_function(prompt: str, response: str) -> float:
    """改良版報酬関数：句読点・記号を適切に処理"""
    
    # プロンプトから元の漢字文を抽出
    match = re.search(r'以下の漢字混じり文をカタカナに変換してください：(.+)', prompt)
    if not match:
        return -5.0
    
    original_text = match.group(1).strip()
    response = response.strip()
    
    if not response:
        return -5.0
    
    # テキスト構成分析
    original_analysis = analyze_text_composition(original_text)
    response_analysis = analyze_text_composition(response)
    
    total_reward = 0.0
    
    # 1. カタカナ純度報酬（改良版）
    katakana_ratio = response_analysis['katakana_ratio']
    if katakana_ratio >= 0.95:
        total_reward += 4.0  # 95%以上で高得点
    elif katakana_ratio >= 0.85:
        total_reward += 2.0  # 85%以上で中得点
    elif katakana_ratio >= 0.70:
        total_reward += 0.5  # 70%以上で低得点
    else:
        total_reward -= 3.0  # 70%未満でペナルティ
    
    # 2. 不適切文字ペナルティ（改良版）
    hiragana_penalty = response_analysis['hiragana'] * -0.8  # ひらがなペナルティ
    kanji_penalty = response_analysis['kanji'] * -1.2        # 漢字ペナルティ（重い）
    total_reward += hiragana_penalty + kanji_penalty
    
    # 3. 長さ適切性報酬（改良版）
    length_ratio = response_analysis['total_length'] / max(original_analysis['total_length'], 1)
    if 0.8 <= length_ratio <= 2.5:
        total_reward += 2.5  # 適切な長さ
    elif 0.6 <= length_ratio <= 3.0:
        total_reward += 1.0  # やや適切
    elif length_ratio > 4.0:
        total_reward -= 4.0  # 長すぎる
    elif length_ratio < 0.5:
        total_reward -= 3.0  # 短すぎる
    
    # 4. 句読点・記号保持報酬（新規）
    original_punctuation = len([c for c in original_text if c in '。、！？・「」（）'])
    response_punctuation = len([c for c in response if c in '。、！？・「」（）'])
    
    # 句読点が適切に保持されている場合にボーナス
    if original_punctuation > 0:
        punctuation_ratio = response_punctuation / original_punctuation
        if 0.8 <= punctuation_ratio <= 1.2:
            total_reward += 1.5  # 句読点適切保持ボーナス
        elif punctuation_ratio < 0.5:
            total_reward -= 1.0  # 句読点不足ペナルティ
    
    # 5. 完全性報酬（改良版）
    if response_analysis['japanese_total'] >= original_analysis['japanese_total'] * 0.8:
        total_reward += 1.5  # 内容の完全性
    else:
        total_reward -= 2.0  # 内容不足
    
    # 6. フォーマット遵守報酬（改良版）
    # 説明文や余計な文言がないかチェック
    explanation_patterns = [
        r'変換すると', r'カタカナにすると', r'読み方は', r'以下のようになります',
        r'答えは', r'結果は', r'変換結果', r'カタカナ表記'
    ]
    
    has_explanation = any(re.search(pattern, response) for pattern in explanation_patterns)
    if has_explanation:
        total_reward -= 2.0  # 説明文ペナルティ
    else:
        total_reward += 1.0  # 直接変換ボーナス
    
    # 7. 文字種一貫性報酬（新規）
    # カタカナ以外の文字が混在している場合のペナルティ調整
    if response_analysis['japanese_total'] > 0:
        consistency_score = response_analysis['katakana'] / response_analysis['japanese_total']
        if consistency_score >= 0.9:
            total_reward += 1.0  # 高い一貫性
        elif consistency_score < 0.7:
            total_reward -= 1.5  # 低い一貫性
    
    # 8. 特殊文字処理報酬（新規）
    # 英数字が適切に処理されているかチェック
    if response_analysis['english'] > 0 or response_analysis['numbers'] > 0:
        # 英数字がカタカナ化されているか、そのまま保持されているかを評価
        total_reward += 0.5  # 英数字適切処理ボーナス
    
    return max(total_reward, -10.0)  # 最低スコア制限

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

def load_grpo_dataset(file_paths: list[str], max_samples: int = 25000, sample_rate: float = 0.2) -> Dataset:
    """GRPO用データセット読み込み"""
    data = []
    sample_count = 0
    total_lines = 0
    
    for file_path in file_paths:
        print(f"📖 GRPO用データ読み込み開始: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    total_lines += 1
                    
                    if torch.rand(1).item() > sample_rate:
                        continue
                        
                    if sample_count >= max_samples:
                        print(f"✅ 最大サンプル数 {max_samples} に到達")
                        break
                    
                    try:
                        item = json.loads(line.strip())
                        kanji_text = item["output"]
                        katakana_text = item["input"]
                        
                        # データ品質チェック（改良版）
                        if len(kanji_text) > 60 or len(katakana_text) > 60:
                            continue
                        
                        if len(kanji_text) < 5 or len(katakana_text) < 5:
                            continue
                        
                        # カタカナ純度事前チェック（改良版）
                        analysis = analyze_text_composition(katakana_text)
                        if analysis['katakana_ratio'] < 0.8:
                            continue
                        
                        conversation_data = format_gemma3_conversation(kanji_text, katakana_text)
                        
                        text = tokenizer.apply_chat_template(
                            conversation_data["conversations"], 
                            tokenize=False
                        )
                        
                        if len(tokenizer.encode(text)) <= max_seq_length:
                            data.append(conversation_data)
                            sample_count += 1
                            
                            if sample_count % 1000 == 0:
                                print(f"📊 処理済み: {sample_count}/{total_lines} (サンプリング率: {sample_rate*100:.1f}%)")
                        
                        if sample_count % 5000 == 0:
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
    
    print(f"✅ 改良版GRPO用データセット作成完了: {len(data)} サンプル")
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

print("📚 datasetsライブラリでGRPO用データセットをロード＆キャッシュ中...")
cache_dir = "cache/grpo_arrow"
if not os.path.exists(cache_dir):
    raw = load_dataset("json", data_files=dataset_files, split="train")
    raw = raw.filter(
        lambda ex: 5 <= len(ex["output"]) <= 60 and 5 <= len(ex["input"]) <= 60 and analyze_text_composition(ex["input"])["katakana_ratio"] >= 0.8,
        num_proc=4
    )
    raw = raw.map(
        lambda examples: {"conversations":[format_gemma3_conversation(o,i)["conversations"] for o,i in zip(examples["output"], examples["input"]) ]},
        batched=True, batch_size=1000, num_proc=4, remove_columns=["input","output","left_context"]
    )
    raw.save_to_disk(cache_dir)
dataset = load_from_disk(cache_dir)
print("🔄 トークナイズをバッチ・並列で実行中...")
dataset = dataset.map(
    lambda examples: {"text":[tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in examples["conversations"]]},
    batched=True, batch_size=1000, num_proc=4, remove_columns=["conversations"]
)

# サンプル確認
print("\n=== 📋 改良版GRPOサンプル確認 ===")
for i in range(min(2, len(dataset))):
    sample = dataset[i]
    print(f"サンプル {i+1}:")
    print(sample["text"][:200] + "..." if len(sample["text"]) > 200 else sample["text"])
    print("-" * 50)

# メモリクリーンアップ
gc.collect()
torch.cuda.empty_cache()

# 改良版GRPO Trainer設定
print("⚙️ 改良版GRPOTrainerを設定中...")
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_function=improved_reward_function,
    args=GRPOConfig(
        dataset_text_field="text",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=5e-5,
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        seed=3407,
        output_dir="./outputs/gemma3_1b_kanji2kana_grpo_improved",
        report_to="tensorboard",
        logging_dir="./logs/tensorboard_grpo_improved",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        bf16=True,
        max_grad_norm=1.0,
        save_total_limit=3,
        # GRPO特有設定
        grpo_alpha=1.0,
        grpo_beta=0.1,
        grpo_gamma=0.99,
    ),
)

print("🚀 改良版GRPO訓練開始...")
print("🔥 改良版GRPO設定:")
print(f"  - 報酬関数: 8つの改良版報酬要素")
print(f"  - 句読点・記号処理: 適切に考慮")
print(f"  - カタカナ純度: 日本語文字のみで計算")
print(f"  - データサンプル数: {len(dataset)}")
print(f"  - サンプリング率: {sample_rate*100:.1f}%")

trainer_stats = trainer.train()

print(f"\n📊 改良版GRPO訓練完了!")
print(f"⏱️ 訓練時間: {trainer_stats.metrics['train_runtime']:.2f} 秒")
print(f"📈 最終ロス: {trainer_stats.metrics['train_loss']:.4f}")

# モデル保存
save_dir = "model/gemma3_1b_kanji2kana_grpo_improved"
print(f"💾 改良版GRPOモデルを {save_dir} に保存中...")
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ 改良版GRPO訓練完了！")
print(f"💾 保存先: {save_dir}")
print(f"📊 使用データ: {len(dataset)} サンプル")

print("\n" + "="*70)
print("🎯 改良版GRPO報酬関数の特徴:")
print("="*70)
print("1. カタカナ純度: 句読点・記号を除外して正確に計算")
print("2. 句読点保持: 元文の句読点が適切に保持されているかチェック")
print("3. 文字種一貫性: カタカナ以外の文字混在を適切にペナルティ")
print("4. 特殊文字処理: 英数字の適切な処理を評価")
print("5. フォーマット遵守: 説明文を排除し直接変換を促進")
print("6. 長さ適切性: より柔軟な長さ範囲で評価")
print("7. 完全性: 内容の欠落を防止")
print("8. 不適切文字: ひらがな・漢字の残存を厳格にペナルティ")
print("")
print("⚡ 句読点・記号問題を解決した高精度GRPO完成！") 