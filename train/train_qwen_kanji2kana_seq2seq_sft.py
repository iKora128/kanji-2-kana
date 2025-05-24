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
# ]
# ///

"""
Qwen1.7BでSFTTrainerを使った漢字→カタカナ変換学習スクリプト（Seq2Seqデータ形式）
"""

import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# モデル設定
max_seq_length = 256  # Seq2Seqでは短めに設定
lora_rank = 16

print("🚀 モデルとトークナイザーを読み込み中...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,  # vLLMを使わない
    max_lora_rank=lora_rank,
)

# Seq2Seqのためのトークナイザー設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRAアダプターを追加
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

def format_seq2seq_prompt(kanji_text: str, katakana_text: str = "") -> str:
    """Seq2Seq形式のプロンプト作成"""
    if katakana_text:
        # 訓練時
        return f"漢字をカタカナに変換してください。\n入力: {kanji_text}\n出力: {katakana_text}"
    else:
        # 推論時
        return f"漢字をカタカナに変換してください。\n入力: {kanji_text}\n出力: "

def load_seq2seq_dataset(file_paths: list[str], max_samples: int = 5000) -> Dataset:
    """Seq2Seq用のデータセットを読み込む"""
    data = []
    sample_count = 0
    
    for file_path in file_paths:
        print(f"📖 読み込み開始: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if sample_count >= max_samples:
                        print(f"✅ 最大サンプル数 {max_samples} に到達")
                        break
                    
                    try:
                        item = json.loads(line.strip())
                        kanji_text = item["output"]      # 漢字混じり文章
                        katakana_text = item["input"]    # カタカナ文章
                        
                        # 長すぎるテキストをスキップ
                        if len(kanji_text) > 80 or len(katakana_text) > 80:
                            continue
                        
                        # 空文字列をスキップ
                        if not kanji_text.strip() or not katakana_text.strip():
                            continue
                        
                        # Seq2Seq形式でフォーマット
                        text = format_seq2seq_prompt(kanji_text, katakana_text)
                        
                        # トークン長チェック
                        if len(tokenizer.encode(text)) <= max_seq_length:
                            data.append({"text": text})
                            sample_count += 1
                            
                            if sample_count % 1000 == 0:
                                print(f"📊 処理済み: {sample_count}")
                    
                    except (json.JSONDecodeError, KeyError):
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
    
    print(f"✅ データセット作成完了: {len(data)} サンプル")
    return Dataset.from_list(data)

# データセット作成
dataset_files = [
    "data/train_llm-jp-corpus-v3.jsonl",
    "data/train_wikipedia.jsonl"
]

print("📚 Seq2Seqデータセットを作成中...")
dataset = load_seq2seq_dataset(dataset_files, max_samples=5000)

# サンプル確認
print("\n=== 📋 サンプル確認 ===")
for i in range(2):
    sample = dataset[i]
    print(f"サンプル {i+1}:")
    print(sample["text"])
    print("-" * 50)

# SFTTrainer設定（Seq2Seqデータ用）
print("⚙️ SFTTrainerを設定中...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        num_train_epochs=10,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./outputs_seq2seq_sft",
        report_to="none",
    ),
)

print("🚀 Seq2Seq SFT訓練開始...")
trainer.train()

# モデル保存
print("💾 モデルを保存中...")
model.save_pretrained("kanji2kana_seq2seq_sft_lora")
tokenizer.save_pretrained("kanji2kana_seq2seq_sft_lora")

# テスト推論
print("\n=== 🧪 テスト推論 ===")
# 推論モードに切り替え
FastLanguageModel.for_inference(model)

def test_inference(text: str) -> str:
    """テスト推論関数"""
    prompt = format_seq2seq_prompt(text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
        padding=True
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
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

print("🔍 推論テスト結果:")
for i, test_case in enumerate(test_cases, 1):
    result = test_inference(test_case)
    print(f"{i}. 入力: {test_case}")
    print(f"   出力: {result}")
    print("-" * 40)

print("\n✅ Seq2Seq SFT訓練完了！")

# メモリ使用量確認
if torch.cuda.is_available():
    print(f"🖥️ GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB") 