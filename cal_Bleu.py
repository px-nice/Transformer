import torch
from transformers import AutoTokenizer
from sacrebleu import corpus_bleu
from tqdm import tqdm
import pickle
import os
from def_model_fixed import Transformer  # 假设这是你的模型定义
import random

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_offline_resources(data_dir="Transformer/test_data"):
    """加载本地保存的所有资源"""
    # 1. 加载分词器
    de_tokenizer = AutoTokenizer.from_pretrained(f"{data_dir}/de_tokenizer")
    en_tokenizer = AutoTokenizer.from_pretrained(f"{data_dir}/en_tokenizer")
    
    # 2. 加载验证文本
    with open(f"{data_dir}/test_texts_full.pkl", 'rb') as f:
        val_pairs = pickle.load(f)
    
    # 3. 加载tokenized样本（备用）
    with open(f"{data_dir}/tokenized_samples.pkl", 'rb') as f:
        tokenized_samples = pickle.load(f)
    
    return de_tokenizer, en_tokenizer, val_pairs, tokenized_samples

def translate(model, src_text, src_tokenizer, tgt_tokenizer, max_length=256):
    """使用本地分词器进行翻译"""
    src_tokens = src_tokenizer.encode(
                src_text,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
                padding='max_length',
                return_tensors='pt'
            ).squeeze(0)
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
    
    tgt_indices = [tgt_tokenizer.cls_token_id]
    for _ in range(max_length):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs, _, _, _ = model(src_tensor, tgt_tensor)
        next_token = torch.argmax(outputs[0,-1, :]).item()
        tgt_indices.append(next_token)
        if next_token == tgt_tokenizer.sep_token_id:
            break
    
    return tgt_tokenizer.decode(tgt_indices[1:-1],skip_special_tokens=True)

def evaluate_bleu_offline(checkpoint_path, data_dir="Transformer/test_data",sample_size=3003):
    # 1. 加载本地资源
    de_tokenizer = AutoTokenizer.from_pretrained(f"{data_dir}/de_tokenizer")
    en_tokenizer = AutoTokenizer.from_pretrained(f"{data_dir}/en_tokenizer")
    
    with open(f"{data_dir}/test_texts.pkl", 'rb') as f:
        text_pairs = pickle.load(f)

    if len(text_pairs) > sample_size:
        text_pairs = random.sample(text_pairs, sample_size)
        print(f"随机抽取{sample_size}条样本进行评估")
    else:
        print(f"样本总数不足{sample_size}条，使用全部{len(text_pairs)}条进行评估")
    
    # 2. 加载模型和checkpoint
    model = Transformer().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. 生成翻译
    hypotheses = []
    references = []
    
    for pair in tqdm(text_pairs, desc="生成翻译"):
        # 英语→德语翻译（根据实际模型方向调整）
        translation = translate(
            model=model,
            src_text=pair['en'],  # 假设英语是源语言
            src_tokenizer=en_tokenizer,
            tgt_tokenizer=de_tokenizer
        )
        hypotheses.append(translation)
        ref_tokens = de_tokenizer.encode(
            pair['de'],
            add_special_tokens=False
        )
        processed_ref = de_tokenizer.decode(ref_tokens,skip_special_tokens=True)
        references.append([processed_ref])  # 注意保持双层列表结构
    
    # 4. 计算BLEU
    bleu_score = corpus_bleu(hypotheses, references)
    print(f"BLEU分数: {bleu_score.score:.2f}")

if __name__ == "__main__":
    # 使用checkpoint路径而不是模型路径
    evaluate_bleu_offline("checkpoints_regularized2/checkpoint_epoch_1_step_150000.pt")