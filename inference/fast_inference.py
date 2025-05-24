# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "torch",
#     "transformers",
#     "vllm",
# ]
# ///

"""
🚀 超高速漢字→カタカナ変換推論スクリプト
最適化された生成パラメータで高速推論
"""

import torch
import time
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from vllm import LLM

class FastKanji2Katakana:
    """高速漢字→カタカナ変換クラス"""
    
    def __init__(self, model_path: str = "model/gemma3_1b_kanji2kana_sft_merged"):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        
    def load_model(self):
        """高速化設定でモデルを読み込み"""
        print(f"🚀 高速推論用モデル読み込み中: {self.model_path}")
        
        start_time = time.time()
        
        # 高速化設定
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=256,  # 短縮（カタカナ変換に十分）
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )
        
        self.tokenizer = get_chat_template(self.tokenizer, chat_template="gemma-3")
        FastModel.for_inference(self.model)
        
        # モデルコンパイル（PyTorch 2.0+で高速化）
        if hasattr(torch, 'compile'):
            print("🔥 モデルをコンパイル中（初回のみ時間がかかります）...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        load_time = time.time() - start_time
        print(f"✅ 高速モデル読み込み完了！ ({load_time:.2f}秒)")
        
        return self
    
    def convert(self, text: str) -> tuple[str, float]:
        """超高速変換"""
        if self.model is None:
            raise RuntimeError("モデルが読み込まれていません。load_model()を先に実行してください。")
        
        start_time = time.time()
        
        messages = [{
            "role": "user",
            "content": f"以下の漢字混じり文をカタカナに変換してください：{text}"
        }]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,          # 大幅短縮（カタカナ変換に十分）
                temperature=0.0,            # 決定的生成（最高速）
                do_sample=False,            # サンプリング無効化（高速化）
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,             # キャッシュ使用
            )
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        return result.strip(), inference_time
    
    def batch_convert(self, texts: list[str]) -> list[tuple[str, float]]:
        """バッチ処理で更なる高速化"""
        results = []
        
        # バッチサイズ調整（GPU メモリに応じて）
        batch_size = min(8, len(texts))
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_start = time.time()
            
            # バッチプロンプト作成
            batch_messages = []
            for text in batch_texts:
                messages = [{
                    "role": "user",
                    "content": f"以下の漢字混じり文をカタカナに変換してください：{text}"
                }]
                batch_messages.append(messages)
            
            # バッチトークン化
            batch_prompts = [
                self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in batch_messages
            ]
            
            batch_inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=256
            ).to("cuda")
            
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=50,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            batch_time = time.time() - batch_start
            
            # 結果をデコード
            for j, (text, output) in enumerate(zip(batch_texts, batch_outputs)):
                input_length = len(batch_inputs["input_ids"][j])
                generated_tokens = output[input_length:]
                result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # 個別の推論時間を推定
                individual_time = batch_time / len(batch_texts)
                results.append((result.strip(), individual_time))
        
        return results

def analyze_text_fast(text: str) -> dict:
    """高速文字種分析"""
    katakana = sum(1 for c in text if '\u30A0' <= c <= '\u30FF')
    hiragana = sum(1 for c in text if '\u3040' <= c <= '\u309F')
    kanji = sum(1 for c in text if '\u4E00' <= c <= '\u9FAF')
    japanese_total = katakana + hiragana + kanji + sum(1 for c in text if c.isalnum())
    
    return {
        'katakana': katakana,
        'hiragana': hiragana,
        'kanji': kanji,
        'japanese_total': japanese_total,
        'katakana_ratio': katakana / max(japanese_total, 1)
    }

def interactive_fast_mode():
    """高速対話モード"""
    print("🚀 超高速漢字→カタカナ変換システム")
    print("="*60)
    
    converter = FastKanji2Katakana().load_model()
    
    # GPU使用量表示
    if torch.cuda.is_available():
        print(f"🖥️ GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print("\n💡 超高速モード設定:")
    print("  - 最大トークン数: 50 (通常の半分)")
    print("  - 決定的生成: temperature=0.0")
    print("  - サンプリング無効: do_sample=False")
    print("  - キャッシュ最適化: 有効")
    print("  - モデルコンパイル: 有効")
    
    print("\n" + "="*60)
    print("🎌 超高速漢字→カタカナ変換")
    print("="*60)
    print("💡 'batch' でバッチモード, 'quit' で終了")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n📝 変換したい文章: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 終了します")
                break
            
            if user_input.lower() == 'batch':
                batch_mode(converter)
                continue
            
            if not user_input:
                continue
            
            # 超高速変換
            result, inference_time = converter.convert(user_input)
            analysis = analyze_text_fast(result)
            
            print(f"📥 出力: {result}")
            print(f"⚡ 推論時間: {inference_time:.3f}秒")
            print(f"🚀 処理速度: {len(user_input)/inference_time:.1f} 文字/秒")
            print(f"📊 カタカナ率: {analysis['katakana_ratio']:.1%} ({analysis['katakana']}/{analysis['japanese_total']}文字)")
            
            if analysis['hiragana'] > 0:
                print(f"⚠️ ひらがな: {analysis['hiragana']}文字")
            if analysis['kanji'] > 0:
                print(f"⚠️ 漢字: {analysis['kanji']}文字")
            
        except KeyboardInterrupt:
            print("\n👋 終了します")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")

def batch_mode(converter):
    """バッチ処理モード"""
    print("\n📦 超高速バッチ処理モード")
    print("="*40)
    print("複数の文章を一度に高速処理します")
    print("空行で処理開始、'back'で戻る")
    print("-" * 40)
    
    texts = []
    while True:
        try:
            line = input().strip()
            if line.lower() == 'back':
                return
            if not line:
                if texts:
                    break
                continue
            texts.append(line)
            print(f"  {len(texts)}. {line}")
        except KeyboardInterrupt:
            return
    
    print(f"\n🚀 {len(texts)}件をバッチ処理中...")
    
    start_time = time.time()
    results = converter.batch_convert(texts)
    total_time = time.time() - start_time
    
    total_chars = sum(len(text) for text in texts)
    
    print(f"\n📊 バッチ処理結果 (総時間: {total_time:.3f}秒)")
    print(f"🚀 総合処理速度: {total_chars/total_time:.1f} 文字/秒")
    print("="*50)
    
    for i, (text, (result, individual_time)) in enumerate(zip(texts, results), 1):
        analysis = analyze_text_fast(result)
        print(f"\n{i}. 入力: {text}")
        print(f"   出力: {result}")
        print(f"   時間: {individual_time:.3f}秒 | 速度: {len(text)/individual_time:.1f} 文字/秒")
        print(f"   カタカナ率: {analysis['katakana_ratio']:.1%}")

def speed_benchmark():
    """推論速度ベンチマーク"""
    print("🏁 推論速度ベンチマーク")
    print("="*50)
    
    converter = FastKanji2Katakana().load_model()
    
    test_texts = [
        "今日は良い天気です。",
        "私は東京駅に行きます。",
        "美しい花が咲いています。",
        "科学技術の発展が著しい。",
        "人工知能が普及している。",
        "コンピューターを使って研究を行う。",
        "新幹線で大阪から東京まで移動した。",
        "国際会議が来年の春に開催される予定です。",
        "機械学習とディープラーニングの技術が急速に進歩している。",
        "自然言語処理の分野では、大規模言語モデルが注目されている。"
    ]
    
    print(f"📝 {len(test_texts)}件の文章でベンチマーク実行...")
    
    # 個別処理ベンチマーク
    print("\n🔥 個別処理ベンチマーク:")
    individual_times = []
    total_chars = 0
    
    for i, text in enumerate(test_texts, 1):
        result, inference_time = converter.convert(text)
        individual_times.append(inference_time)
        total_chars += len(text)
        
        print(f"{i:2d}. {len(text):2d}文字 | {inference_time:.3f}秒 | {len(text)/inference_time:5.1f} 文字/秒 | {text[:20]}...")
    
    print(f"\n📊 個別処理統計:")
    print(f"平均時間: {sum(individual_times)/len(individual_times):.3f}秒")
    print(f"総合速度: {total_chars/sum(individual_times):.1f} 文字/秒")
    
    # バッチ処理ベンチマーク
    print(f"\n🚀 バッチ処理ベンチマーク:")
    batch_start = time.time()
    batch_results = converter.batch_convert(test_texts)
    batch_total_time = time.time() - batch_start
    
    print(f"バッチ総時間: {batch_total_time:.3f}秒")
    print(f"バッチ総合速度: {total_chars/batch_total_time:.1f} 文字/秒")
    print(f"速度向上率: {(total_chars/batch_total_time) / (total_chars/sum(individual_times)):.1f}x")

def hf_pipeline_mode():
    """Transformers pipelineによる高速推論モード"""
    print("🚀 HF pipelineによる推論モード")
    tokenizer = AutoTokenizer.from_pretrained(
        "model/gemma3_1b_kanji2kana_sft_merged", use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "model/gemma3_1b_kanji2kana_sft_merged",
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=30,
        do_sample=False,
    )
    while True:
        text = input("📝 文章を入力 (quitで終了): ").strip()
        if text.lower() in ['quit','exit','q']:
            break
        prompt = f"以下の漢字混じり文をカタカナに変換してください：{text}"
        res = generator(prompt)[0]["generated_text"]
        # プロンプト部分を削除
        output = res.split(prompt)[-1]
        print(f"📥 出力: {output.strip()}")

def vllm_mode():
    """vLLMによる高速推論モード"""
    print("🚀 vLLMによる推論モード")
    llm = LLM(
        model="model/gemma3_1b_kanji2kana_sft_merged",
        dtype="bfloat16",
        tensor_parallel_size=1
    )
    while True:
        text = input("📝 文章を入力 (quitで終了): ").strip()
        if text.lower() in ['quit','exit','q']:
            break
        prompt = f"以下の漢字混じり文をカタカナに変換してください：{text}"
        outputs = llm.generate([prompt], max_tokens=30)
        result = outputs[0].text
        # プロンプト部分を削除
        output = result.split(prompt)[-1]
        print(f"📥 出力: {output.strip()}")

def main():
    """メイン関数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--benchmark':
            speed_benchmark()
        elif sys.argv[1] == '--hf':
            hf_pipeline_mode()
        elif sys.argv[1] == '--vllm':
            vllm_mode()
        elif sys.argv[1] in ['--help','-h']:
            print("使い方:")
            print("  python fast_inference.py                # 超高速対話モード")
            print("  python fast_inference.py --benchmark    # 速度ベンチマーク")
            print("  python fast_inference.py --hf           # Transformers pipelineモード")
            print("  python fast_inference.py --vllm         # vLLMモード")
        else:
            print("不明なオプション:", sys.argv[1])
    else:
        interactive_fast_mode()

if __name__ == "__main__":
    main() 