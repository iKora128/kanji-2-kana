# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "torch",
#     "transformers",
# ]
# ///

"""
🚀 Gemma-3-1B漢字→カタカナ変換推論スクリプト
学習済みSFTモデルを使用した高速推論
"""

import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import sys
import os
import time

def load_trained_model(model_path: str = "model/gemma3_1b_kanji2kana_sft_merged"):
    """学習済みモデルを読み込む（mergedバージョンで高速推論）"""
    print(f"🚀 最適化済みモデルを読み込み中: {model_path}")
    
    try:
        # mergedモデルを読み込み（推論最適化済み）
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )
        
        # チャットテンプレート設定
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )
        
        # 推論モードに設定
        FastModel.for_inference(model)
        
        print("✅ モデル読み込み完了！")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        print(f"💡 パス '{model_path}' を確認してください")
        return None, None

def kanji_to_katakana(model, tokenizer, text: str) -> tuple[str, float]:
    """漢字混じり文をカタカナに変換（推論時間も測定）"""
    start_time = time.time()
    
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
            temperature=0.1,                # 決定的な生成
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,
        )
    
    # 入力部分を除去して出力のみ取得
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    inference_time = time.time() - start_time
    return result.strip(), inference_time

def interactive_mode(model, tokenizer):
    """対話モード"""
    print("\n" + "="*60)
    print("🎌 漢字→カタカナ変換 対話モード")
    print("="*60)
    print("💡 使い方:")
    print("  - 漢字混じりの文章を入力してください")
    print("  - 'quit' または 'exit' で終了")
    print("  - 'batch' でバッチモードに切り替え")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n📝 変換したい文章を入力: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 推論を終了します")
                break
            
            if user_input.lower() == 'batch':
                batch_mode(model, tokenizer)
                continue
            
            if not user_input:
                print("⚠️ 文章を入力してください")
                continue
            
            # 変換実行
            print("🔄 変換中...")
            result, inference_time = kanji_to_katakana(model, tokenizer, user_input)
            
            print(f"📤 入力: {user_input}")
            print(f"📥 出力: {result}")
            print(f"⏱️ 推論時間: {inference_time:.3f}秒")
            
            # 詳細評価（句読点・記号を除外）
            katakana_chars = len([c for c in result if '\u30A0' <= c <= '\u30FF'])
            # 日本語文字のみをカウント（ひらがな、カタカナ、漢字、英数字）
            japanese_chars = len([c for c in result if (
                '\u3040' <= c <= '\u309F' or  # ひらがな
                '\u30A0' <= c <= '\u30FF' or  # カタカナ
                '\u4E00' <= c <= '\u9FAF' or  # 漢字
                c.isalnum()                   # 英数字
            )])
            katakana_ratio = katakana_chars / max(japanese_chars, 1)
            
            print(f"📊 カタカナ率: {katakana_ratio:.1%}")
            print(f"🚀 推論速度: {len(user_input)/inference_time:.1f} 文字/秒")
            
        except KeyboardInterrupt:
            print("\n👋 推論を終了します")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")

def batch_mode(model, tokenizer):
    """バッチ処理モード"""
    print("\n" + "="*60)
    print("📦 バッチ処理モード")
    print("="*60)
    print("💡 複数の文章を一度に処理します")
    print("  - 各行に1つの文章を入力")
    print("  - 空行で処理開始")
    print("  - 'back' で対話モードに戻る")
    print("-" * 60)
    
    texts = []
    print("📝 変換したい文章を入力してください (空行で処理開始):")
    
    while True:
        try:
            line = input().strip()
            
            if line.lower() == 'back':
                return
            
            if not line:  # 空行で処理開始
                if texts:
                    break
                else:
                    print("⚠️ 文章を入力してください")
                    continue
            
            texts.append(line)
            print(f"  {len(texts)}. {line}")
            
        except KeyboardInterrupt:
            return
    
    # バッチ処理実行
    print(f"\n🔄 {len(texts)}件の文章を変換中...")
    results = []
    
    total_time = 0
    for i, text in enumerate(texts, 1):
        try:
            result, inference_time = kanji_to_katakana(model, tokenizer, text)
            total_time += inference_time
            results.append((text, result, inference_time))
            print(f"✅ {i}/{len(texts)} 完了 ({inference_time:.3f}秒)")
        except Exception as e:
            results.append((text, f"エラー: {e}", 0))
            print(f"❌ {i}/{len(texts)} エラー")
    
    # 結果表示
    print("\n" + "="*60)
    print("📋 バッチ処理結果")
    print("="*60)
    print(f"📊 総処理時間: {total_time:.3f}秒")
    print(f"⚡ 平均処理時間: {total_time/len(texts):.3f}秒/文")
    print(f"🚀 平均処理速度: {sum(len(t) for t, _, _ in results)/total_time:.1f} 文字/秒")
    
    for i, (input_text, output_text, inference_time) in enumerate(results, 1):
        print(f"\n{i}. 入力: {input_text}")
        print(f"   出力: {output_text}")
        if inference_time > 0:
            print(f"   推論時間: {inference_time:.3f}秒")
        
        if not output_text.startswith("エラー:"):
            katakana_chars = len([c for c in output_text if '\u30A0' <= c <= '\u30FF'])
            # 日本語文字のみをカウント（句読点・記号を除外）
            japanese_chars = len([c for c in output_text if (
                '\u3040' <= c <= '\u309F' or  # ひらがな
                '\u30A0' <= c <= '\u30FF' or  # カタカナ
                '\u4E00' <= c <= '\u9FAF' or  # 漢字
                c.isalnum()                   # 英数字
            )])
            katakana_ratio = katakana_chars / max(japanese_chars, 1)
            
            # 文字種別詳細分析
            hiragana_chars = len([c for c in output_text if '\u3040' <= c <= '\u309F'])
            kanji_chars = len([c for c in output_text if '\u4E00' <= c <= '\u9FAF'])
            
            print(f"   カタカナ率: {katakana_ratio:.1%} ({katakana_chars}/{japanese_chars}文字)")
            if hiragana_chars > 0:
                print(f"   ⚠️ ひらがな: {hiragana_chars}文字")
            if kanji_chars > 0:
                print(f"   ⚠️ 漢字: {kanji_chars}文字")
            if inference_time > 0:
                print(f"   処理速度: {len(input_text)/inference_time:.1f} 文字/秒")
        
        print("-" * 40)

def file_mode(model, tokenizer, input_file: str, output_file: str):
    """ファイル処理モード"""
    print(f"📂 ファイル処理モード: {input_file} → {output_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"📖 {len(lines)}行を読み込みました")
        
        results = []
        total_time = 0
        for i, line in enumerate(lines, 1):
            try:
                result, inference_time = kanji_to_katakana(model, tokenizer, line)
                total_time += inference_time
                results.append((result, inference_time))
                print(f"✅ {i}/{len(lines)} 完了: {line[:30]}... ({inference_time:.3f}秒)")
            except Exception as e:
                results.append((f"エラー: {e}", 0))
                print(f"❌ {i}/{len(lines)} エラー: {line[:30]}...")
        
        # 結果をファイルに保存
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"📊 総処理時間: {total_time:.3f}秒\n")
            f.write(f"⚡ 平均処理時間: {total_time/len(lines):.3f}秒/文\n")
            f.write(f"🚀 平均処理速度: {sum(len(line) for line in lines)/total_time:.1f} 文字/秒\n")
            f.write("=" * 60 + "\n\n")
            
            for original, (converted, inference_time) in zip(lines, results):
                f.write(f"入力: {original}\n")
                f.write(f"出力: {converted}\n")
                if inference_time > 0:
                    f.write(f"推論時間: {inference_time:.3f}秒\n")
                    f.write(f"処理速度: {len(original)/inference_time:.1f} 文字/秒\n")
                f.write("-" * 40 + "\n")
        
        print(f"✅ 結果を {output_file} に保存しました")
        print(f"📊 総処理時間: {total_time:.3f}秒")
        print(f"🚀 平均処理速度: {sum(len(line) for line in lines)/total_time:.1f} 文字/秒")
        
    except Exception as e:
        print(f"❌ ファイル処理エラー: {e}")

def main():
    """メイン関数"""
    print("🚀 Gemma-3-1B 漢字→カタカナ変換推論システム")
    print("="*60)
    
    # コマンドライン引数チェック
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("使い方:")
            print("  python inference_gemma3_kanji2kana.py                    # 対話モード")
            print("  python inference_gemma3_kanji2kana.py input.txt output.txt  # ファイルモード")
            print("  python inference_gemma3_kanji2kana.py --model path/to/model  # カスタムモデル")
            return
    
    # モデルパス設定（mergedバージョンで高速推論）
    model_path = "model/gemma3_1b_kanji2kana_sft_merged"
    
    # カスタムモデルパス指定
    if len(sys.argv) >= 3 and sys.argv[1] == '--model':
        model_path = sys.argv[2]
        print(f"📂 カスタムモデルパス: {model_path}")
    
    # モデル読み込み
    model, tokenizer = load_trained_model(model_path)
    if model is None:
        return
    
    # GPU使用量確認
    if torch.cuda.is_available():
        print(f"🖥️ GPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 実行モード判定
    if len(sys.argv) == 3 and sys.argv[1] != '--model':
        # ファイルモード
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        file_mode(model, tokenizer, input_file, output_file)
    else:
        # 対話モード
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main() 