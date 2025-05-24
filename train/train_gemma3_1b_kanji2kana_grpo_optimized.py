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
🚀 Gemma-3-1B GRPO最適化版 - L4 23GBメモリを最大限活用
カタカナ変換タスクでメモリ効率とバッチサイズを大幅改善した強化学習スクリプト
"""

import json
import torch
import os
import re
import gc
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer

# メモリ最適化設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# L4最適化モデル設定
max_seq_length = 256        # カタカナ変換は短文が多いため短縮
max_prompt_length = 128     # プロンプト長も短縮
lora_rank = 32              # LoRAランクを倍増（品質向上）

print("🚀 Gemma-3-1B最適化モデルとトークナイザーを読み込み中...")
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

# ==================== 最適化された報酬関数 ====================

def check_katakana_purity_optimized(completions, **kwargs):
    """
    最適化されたカタカナ純度チェック
    より細かい段階評価とボーナス制度を追加
    """
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"].strip()
        
        if not response:
            scores.append(-3.0)  # 空回答のペナルティを強化
            continue
            
        # カタカナ文字のみかどうかチェック
        katakana_chars = 0
        total_chars = 0
        
        for char in response:
            if char.isspace():
                continue
            total_chars += 1
            if '\u30A0' <= char <= '\u30FF':
                katakana_chars += 1
        
        if total_chars > 0:
            purity = katakana_chars / total_chars
            # より細かい段階評価
            if purity == 1.0:       # 100%カタカナ（完璧）
                score += 4.0
            elif purity >= 0.98:    # 98%以上
                score += 3.5
            elif purity >= 0.95:    # 95%以上
                score += 3.0
            elif purity >= 0.9:     # 90%以上
                score += 2.0
            elif purity >= 0.8:     # 80%以上
                score += 1.0
            elif purity >= 0.6:     # 60%以上
                score += 0.2
            else:
                score -= 2.5  # 強化されたペナルティ
        else:
            score -= 2.0
            
        scores.append(score)
    return scores

def check_length_appropriateness_optimized(prompts, completions, **kwargs):
    """
    最適化された長さ妥当性チェック
    より詳細な範囲設定とボーナス制度
    """
    scores = []
    for prompt, completion in zip(prompts, completions):
        score = 0
        response = completion[0]["content"].strip()
        
        # プロンプトから元の文字を抽出
        user_content = prompt[-1]["content"]
        if "：" in user_content:
            original_text = user_content.split("：")[-1].strip()
        else:
            original_text = user_content.split(":")[-1].strip()
        
        original_length = len(original_text.replace(" ", ""))
        response_length = len(response.replace(" ", ""))
        
        if original_length > 0:
            ratio = response_length / original_length
            
            # より詳細な評価範囲
            if 0.9 <= ratio <= 1.3:    # 理想的な範囲（90-130%）
                score += 3.0
            elif 0.8 <= ratio <= 1.6:  # 良い範囲（80-160%）
                score += 2.0
            elif 0.6 <= ratio <= 2.0:  # 許容範囲（60-200%）
                score += 0.8
            elif 0.4 <= ratio <= 3.0:  # 広い許容範囲（40-300%）
                score += 0.2
            elif ratio > 4.0:          # 極端に長い
                score -= 3.0
            elif ratio < 0.3:          # 極端に短い
                score -= 2.5
            else:
                score -= 1.5
                
        scores.append(score)
    return scores

def check_unwanted_characters_optimized(completions, **kwargs):
    """
    最適化された不要文字チェック
    カタカナ中点・長音符は許可、より厳格なペナルティ
    """
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"].strip()
        
        penalty = 0
        
        # 英語文字（より厳格）
        english_count = len(re.findall(r'[a-zA-Z]', response))
        penalty += english_count * 0.5
        
        # 数字
        number_count = len(re.findall(r'[0-9]', response))
        penalty += number_count * 0.4
        
        # ひらがな（強いペナルティ）
        hiragana_count = len(re.findall(r'[\u3040-\u309F]', response))
        penalty += hiragana_count * 0.8
        
        # 漢字（最も強いペナルティ）
        kanji_count = len(re.findall(r'[\u4e00-\u9faf]', response))
        penalty += kanji_count * 1.2
        
        # 特殊記号（カタカナ中点・長音符は除外）
        allowed_symbols = ['・', 'ー', '　']  # 許可する記号
        for char in response:
            if char in '!@#$%^&*()_+=[]{}|;:<>?/~`"\'\\':
                if char not in allowed_symbols:
                    penalty += 0.3
        
        score -= penalty
        
        # 完全に空の場合
        if not response:
            score -= 3.0
            
        scores.append(score)
    return scores

def check_output_completeness_optimized(completions, **kwargs):
    """
    最適化された完全性チェック
    より細かい長さ評価と改行・空白の処理
    """
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"].strip()
        
        response_length = len(response)
        
        # より細かい長さ評価
        if response_length == 0:
            score -= 4.0
        elif 1 <= response_length <= 2:
            score -= 2.0
        elif 3 <= response_length <= 5:
            score -= 0.5
        elif 6 <= response_length <= 80:   # 理想的な範囲
            score += 2.0
        elif 81 <= response_length <= 120: # 少し長いが許容
            score += 1.0
        elif 121 <= response_length <= 200: # 長い
            score -= 0.8
        else:  # 極端に長い
            score -= 2.5
        
        # 改行数のチェック（より厳格）
        newline_count = response.count('\n')
        if newline_count > 1:
            score -= newline_count * 0.5
            
        # 連続空白のペナルティ
        if '  ' in response:  # 連続空白
            score -= 0.3
            
        scores.append(score)
    return scores

def check_format_compliance_optimized(completions, **kwargs):
    """
    最適化されたフォーマット準拠チェック
    より包括的な不要フレーズ検出とカタカナ純度ボーナス
    """
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"].strip()
        
        # 拡張された不要フレーズリスト
        unwanted_phrases = [
            "変換すると", "カタカナで", "答えは", "結果は", "以下", "次のように",
            "変換結果", "答え", "回答", "です", "である", "ます", "変換", "して",
            "という", "になります", "は", "が", "を", "に", "の", "と", "だ",
            "れる", "られる", "される", "できる", "思う", "考える", "でしょう"
        ]
        
        penalty = 0
        for phrase in unwanted_phrases:
            penalty += response.count(phrase) * 0.4
        
        score -= penalty
        
        # カタカナ純度による大幅ボーナス
        if len(response) > 0:
            katakana_chars = len([c for c in response if '\u30A0' <= c <= '\u30FF'])
            total_chars = len([c for c in response if not c.isspace()])
            
            if total_chars > 0:
                purity = katakana_chars / total_chars
                if purity == 1.0:
                    score += 2.5  # 完全カタカナボーナス
                elif purity >= 0.95:
                    score += 1.8
                elif purity >= 0.9:
                    score += 1.2
                    
        # 簡潔性ボーナス
        if 3 <= len(response) <= 30 and response.count(' ') <= 2:
            score += 0.8
            
        scores.append(score)
    return scores

def check_semantic_appropriateness(prompts, completions, **kwargs):
    """
    新しい報酬関数：意味的妥当性チェック
    明らかに不適切な変換にペナルティ
    """
    scores = []
    for prompt, completion in zip(prompts, completions):
        score = 0
        response = completion[0]["content"].strip()
        
        # 元のテキストを取得
        user_content = prompt[-1]["content"]
        if "：" in user_content:
            original_text = user_content.split("：")[-1].strip()
        else:
            original_text = user_content.split(":")[-1].strip()
        
        # 基本的な意味的チェック
        if response:
            # 繰り返し文字のペナルティ
            for char in set(response):
                if response.count(char) > len(original_text):
                    score -= 0.5
            
            # 単調な繰り返しパターンのペナルティ
            if len(set(response.replace(' ', ''))) < 3 and len(response) > 5:
                score -= 1.5
                
            # 極端に単純すぎる変換のペナルティ
            if len(response.replace(' ', '')) < 3 and len(original_text) > 8:
                score -= 1.0
                
        scores.append(score)
    return scores

# ==================== 最適化されたデータセット準備 ====================

def format_grpo_prompt(kanji_text: str) -> list[dict]:
    """GRPO用のプロンプト形式"""
    return [
        {
            "role": "user",
            "content": f"以下の漢字混じり文をカタカナに変換してください：{kanji_text}"
        }
    ]

def load_optimized_grpo_dataset(file_paths: list[str], max_samples: int = 25000) -> Dataset:
    """
    最適化されたGRPO用データセット読み込み
    より多くのサンプルと効率的な処理
    """
    data = []
    sample_count = 0
    
    for file_path in file_paths:
        print(f"📖 大容量データ読み込み開始: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if sample_count >= max_samples:
                        break
                    
                    # サンプリング率を上げて高品質データを多く使用
                    if torch.rand(1).item() > 0.2:  # 20%をサンプリング（4倍増）
                        continue
                    
                    try:
                        item = json.loads(line.strip())
                        kanji_text = item["output"]      # 漢字混じり文章
                        katakana_text = item["input"]    # カタカナ文章
                        
                        # より厳格なデータ品質チェック
                        if len(kanji_text) > 40 or len(katakana_text) > 40:
                            continue
                        
                        if len(kanji_text) < 3 or len(katakana_text) < 3:
                            continue
                        
                        if not kanji_text.strip() or not katakana_text.strip():
                            continue
                        
                        # 文字種の多様性チェック
                        if len(set(kanji_text)) < 3:  # 文字の種類が少なすぎる
                            continue
                            
                        # カタカナ純度事前チェック
                        katakana_chars = len([c for c in katakana_text if '\u30A0' <= c <= '\u30FF'])
                        total_chars = len([c for c in katakana_text if not c.isspace()])
                        if total_chars > 0 and katakana_chars / total_chars < 0.8:
                            continue
                        
                        prompt_data = {
                            "prompt": format_grpo_prompt(kanji_text),
                            "kanji_text": kanji_text,
                            "expected_katakana": katakana_text,
                            "original_length": len(kanji_text),
                            "target_length": len(katakana_text)
                        }
                        
                        data.append(prompt_data)
                        sample_count += 1
                        
                        if sample_count % 500 == 0:
                            print(f"📊 高品質データ処理済み: {sample_count}")
                        
                        # メモリ管理
                        if sample_count % 5000 == 0:
                            gc.collect()
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        continue
                        
            if sample_count >= max_samples:
                break
                
        except FileNotFoundError:
            print(f"❌ ファイルが見つかりません: {file_path}")
            continue
    
    print(f"✅ 最適化GRPO用データセット作成完了: {len(data)} サンプル")
    return Dataset.from_list(data)

# データセット作成
dataset_files = [
    "data/train_llm-jp-corpus-v3.jsonl",
    "data/train_wikipedia.jsonl"
]

print("📚 大容量最適化GRPO用データセットを作成中...")
dataset = load_optimized_grpo_dataset(dataset_files, max_samples=25000)

# サンプル確認
print("\n=== 📋 最適化GRPO用サンプル確認 ===")
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    print(f"サンプル {i+1}:")
    print(f"プロンプト: {sample['prompt']}")
    print(f"期待出力: {sample['expected_katakana']}")
    print(f"長さ比率: {sample['original_length']} → {sample['target_length']}")
    print("-" * 50)

# メモリクリーンアップ
gc.collect()
torch.cuda.empty_cache()

# ==================== L4最適化GRPO訓練設定 ====================

print("⚙️ L4最大活用GRPOTrainerを設定中...")

training_args = GRPOConfig(
    learning_rate=3e-6,                    # 大きなバッチサイズに合わせて学習率調整
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.05,                     # 過学習防止を強化
    warmup_ratio=0.05,                     # 短いウォームアップ
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    logging_steps=2,                       # より頻繁なログ
    per_device_train_batch_size=16,        # 大幅増加（元の8倍）
    gradient_accumulation_steps=4,         # 実効バッチサイズ = 64
    num_generations=8,                     # 候補数を倍増（品質向上）
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=1000,                        # ステップ数大幅増加
    save_steps=100,                        # より頻繁な保存
    eval_steps=50,                         # 評価ステップ追加
    max_grad_norm=0.5,                     # 勾配クリッピング緩和
    report_to="tensorboard",
    output_dir="outputs/gemma3_1b_kanji2kana_grpo_optimized",
    logging_dir="logs/tensorboard_grpo_optimized",
    dataloader_pin_memory=True,            # データローダー最適化
    dataloader_num_workers=4,              # 並列処理数
    remove_unused_columns=False,           # メタデータ保持
    bf16=True,                             # より効率的な計算
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        check_katakana_purity_optimized,        # 最重要：最適化されたカタカナ純度
        check_length_appropriateness_optimized, # 最適化された長さ妥当性
        check_unwanted_characters_optimized,    # 最適化された不要文字チェック
        check_output_completeness_optimized,    # 最適化された完全性チェック
        check_format_compliance_optimized,      # 最適化されたフォーマット準拠
        check_semantic_appropriateness,         # 新規：意味的妥当性チェック
    ],
    args=training_args,
    train_dataset=dataset,
)

print("🚀 L4最大活用 Gemma-3-1B GRPO訓練開始...")
print("📊 最適化された報酬関数（6個）:")
print("  1. カタカナ純度（最適化版） - 100%で+4.0, より細かい段階評価")
print("  2. 長さ妥当性（最適化版） - 90-130%で+3.0, より詳細な範囲")
print("  3. 不要文字（最適化版） - より厳格なペナルティ, カタカナ記号許可")
print("  4. 完全性（最適化版） - より細かい長さ評価, 改行・空白処理")
print("  5. フォーマット準拠（最適化版） - 拡張不要フレーズ, カタカナ純度ボーナス")
print("  6. 意味的妥当性（新規） - 繰り返し・単調パターンのペナルティ")
print("")
print("🔥 L4最適化設定:")
print(f"  - バッチサイズ: {training_args.per_device_train_batch_size} (元の8倍)")
print(f"  - 実効バッチサイズ: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - 候補生成数: {training_args.num_generations} (元の2倍)")
print(f"  - データサンプル数: {len(dataset)} (元の5倍)")
print(f"  - 訓練ステップ数: {training_args.max_steps} (元の5倍)")

# メモリ使用量確認
if torch.cuda.is_available():
    print(f"🖥️ 訓練前GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

trainer_stats = trainer.train()

# 訓練統計表示
print(f"\n📊 最適化GRPO訓練完了!")
print(f"⏱️ 訓練時間: {trainer_stats.metrics['train_runtime']:.2f} 秒")

# モデル保存
save_dir = "model/gemma3_1b_kanji2kana_grpo_optimized"
print(f"💾 最適化モデルを {save_dir} に保存中...")
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# float16形式でも保存
print("💾 float16形式でも保存中...")
model.save_pretrained_merged(f"{save_dir}_merged", tokenizer)

# ==================== 最適化テスト推論 ====================

print("\n=== 🧪 最適化GRPO訓練後テスト推論 ===")
FastModel.for_inference(model)

def test_optimized_grpo_inference(text: str) -> str:
    """最適化GRPO訓練後のテスト推論"""
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
            max_new_tokens=80,
            temperature=0.2,    # より決定的な生成
            top_p=0.85,         # より集中した候補選択
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,  # 繰り返し防止
        )
    
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

print("🔍 最適化GRPO訓練後の推論テスト結果:")
for i, test_case in enumerate(test_cases, 1):
    try:
        result = test_optimized_grpo_inference(test_case)
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

print(f"\n✅ L4最適化 Gemma-3-1B Kanji2Kana GRPO訓練完了！")
print(f"💾 保存先: {save_dir}")
print(f"📊 使用データ: {len(dataset)} サンプル")

print("\n" + "="*70)
print("🚀 L4最適化GRPO の詳細改善点:")
print("="*70)
print("【メモリ活用】")
print("- バッチサイズ: 2 → 16 (8倍増)")
print("- 実効バッチサイズ: 4 → 64 (16倍増)")
print("- シーケンス長: 512 → 256 (効率化)")
print("- LoRAランク: 16 → 32 (品質向上)")
print("")
print("【データ品質】")
print("- サンプル数: 5,000 → 25,000 (5倍増)")
print("- サンプリング率: 5% → 20% (4倍増)")
print("- 品質フィルタリング強化")
print("")
print("【報酬関数強化】")
print("- 6つの最適化された報酬関数")
print("- より細かい段階評価")
print("- 意味的妥当性チェック追加")
print("- カタカナ記号(・、ー)許可")
print("")
print("【学習効率】")
print("- 候補生成数: 4 → 8 (2倍増)")
print("- 訓練ステップ: 200 → 1,000 (5倍増)")
print("- 繰り返し防止強化")
print("")
print("🎯 期待される大幅改善:")
print("- 学習速度: 16倍高速化")
print("- GPUメモリ活用率: 10% → 70-80%")
print("- カタカナ変換精度の大幅向上")
print("- より自然で適切な長さの出力")
print("")
print("⚡ 将来の40GBデータ対応準備完了！") 