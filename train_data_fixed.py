import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm

# 超参数
batch_size = 64
max_length = 256
chunk_size = 10000  # 每处理1万条保存一次

# 加载tokenizer
de_tokenizer = AutoTokenizer.from_pretrained("Transformer/val_data/de_tokenizer")
en_tokenizer = AutoTokenizer.from_pretrained("Transformer/val_data/en_tokenizer")

# 修复：正确获取特殊token ID
def get_special_token_id(tokenizer, token_name):
    """安全地获取特殊token ID"""
    vocab = tokenizer.get_vocab()
    if token_name in vocab:
        return vocab[token_name]
    else:
        # 如果找不到，使用tokenizer的默认特殊token
        if token_name == '[PAD]':
            return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        elif token_name == '[CLS]':
            return tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 101
        elif token_name == '[SEP]':
            return tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 102
        else:
            return 100  # [UNK] token的默认ID

# 获取特殊token ID
PAD_ID = get_special_token_id(de_tokenizer, '[PAD]')
SOS_ID = get_special_token_id(de_tokenizer, '[CLS]')
EOS_ID = get_special_token_id(de_tokenizer, '[SEP]')

print(f"特殊token ID: PAD={PAD_ID}, SOS={SOS_ID}, EOS={EOS_ID}")

# 创建输出目录
os.makedirs('data_chunks_fixed', exist_ok=True)

# 流式加载数据集（避免全量加载到内存）
dataset = load_dataset('wmt14', 'de-en', split='train', streaming=True)

def process_and_save_chunk(chunk, chunk_idx):
    """处理并保存一个数据块"""
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    
    for example in chunk:
        try:
            en_text = example['translation']['en']
            de_text = example['translation']['de']
            
            # 检查文本长度，跳过过长或过短的句子
            if len(en_text.split()) > 100 or len(de_text.split()) > 100:
                continue
            if len(en_text.strip()) < 3 or len(de_text.strip()) < 3:
                continue

            # 修复：正确编码英语输入
            enc_tokens = en_tokenizer.encode(
                en_text,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
                return_tensors='pt'
            ).squeeze(0)
            
            # 如果序列太短，跳过
            if len(enc_tokens) < 2:
                continue
            
            # 修复：正确处理德语序列的teacher forcing格式
            # 先编码德语文本（不添加特殊token）
            de_tokens = de_tokenizer.encode(
                de_text,
                max_length=max_length-2,  # 为特殊token留出空间
                truncation=True,
                add_special_tokens=False
            )
            
            if len(de_tokens) < 1:
                continue
            
            # 修复：手动构建正确的输入输出序列
            # 解码器输入: [SOS] + de_tokens
            dec_input_tokens = [SOS_ID] + de_tokens
            # 解码器输出: de_tokens + [EOS]
            dec_output_tokens = de_tokens + [EOS_ID]
            
            # 确保长度一致
            max_dec_len = max_length
            
            # 填充到固定长度
            if len(enc_tokens) < max_length:
                enc_padded = torch.cat([enc_tokens, torch.zeros(max_length - len(enc_tokens), dtype=torch.long)])
            else:
                enc_padded = enc_tokens[:max_length]
            
            if len(dec_input_tokens) < max_dec_len:
                dec_input_padded = dec_input_tokens + [PAD_ID] * (max_dec_len - len(dec_input_tokens))
            else:
                dec_input_padded = dec_input_tokens[:max_dec_len]
                
            if len(dec_output_tokens) < max_dec_len:
                dec_output_padded = dec_output_tokens + [PAD_ID] * (max_dec_len - len(dec_output_tokens))
            else:
                dec_output_padded = dec_output_tokens[:max_dec_len]
            
            # 转换为tensor
            enc_tensor = enc_padded
            dec_input_tensor = torch.tensor(dec_input_padded, dtype=torch.long)
            dec_output_tensor = torch.tensor(dec_output_padded, dtype=torch.long)
            
            # 验证teacher forcing格式
            # dec_input[1:] 应该等于 dec_output[:-1] (除了padding部分)
            non_pad_len = len([t for t in dec_input_tokens if t != PAD_ID])
            if non_pad_len > 1:
                if not torch.equal(dec_input_tensor[1:non_pad_len], dec_output_tensor[:non_pad_len-1]):
                    print(f"警告: Teacher forcing格式不正确，跳过样本")
                    continue

            enc_inputs.append(enc_tensor)
            dec_inputs.append(dec_input_tensor)
            dec_outputs.append(dec_output_tensor)
            
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue

    if len(enc_inputs) == 0:
        print(f"警告: 块 {chunk_idx} 没有有效样本")
        return

    # 保存当前块
    torch.save(torch.stack(enc_inputs), f'data_chunks_fixed/enc_inputs_{chunk_idx}.pt')
    torch.save(torch.stack(dec_inputs), f'data_chunks_fixed/dec_inputs_{chunk_idx}.pt')
    torch.save(torch.stack(dec_outputs), f'data_chunks_fixed/dec_outputs_{chunk_idx}.pt')
    
    print(f"块 {chunk_idx} 保存完成，包含 {len(enc_inputs)} 个样本")

# 分块处理数据
current_chunk = []
chunk_idx = 0
processed_count = 0

print("开始处理数据...")
for example in tqdm(dataset, desc="处理数据集"):
    current_chunk.append(example)
    processed_count += 1
    
    if len(current_chunk) >= chunk_size:
        process_and_save_chunk(current_chunk, chunk_idx)
        current_chunk = []
        chunk_idx += 1
        
        '''
        # 限制处理数量（用于测试）
        if processed_count >= 50000:  # 只处理5万条数据用于测试
            break
        '''

# 处理剩余数据
if current_chunk:
    process_and_save_chunk(current_chunk, chunk_idx)

print(f"数据处理完成！共处理 {processed_count} 条数据，保存到 data_chunks_fixed/ 目录")

# 验证数据质量
print("\n验证数据质量...")
if chunk_idx >= 0:
    try:
        # 加载第一个块进行验证
        enc_data = torch.load('data_chunks_fixed/enc_inputs_0.pt')
        dec_in_data = torch.load('data_chunks_fixed/dec_inputs_0.pt')
        dec_out_data = torch.load('data_chunks_fixed/dec_outputs_0.pt')
        
        print(f"数据形状验证:")
        print(f"  编码器输入: {enc_data.shape}")
        print(f"  解码器输入: {dec_in_data.shape}")
        print(f"  解码器输出: {dec_out_data.shape}")
        
        # 验证teacher forcing格式
        sample_dec_in = dec_in_data[0]
        sample_dec_out = dec_out_data[0]
        
        # 找到第一个padding位置
        pad_pos_in = (sample_dec_in == PAD_ID).nonzero()
        pad_pos_out = (sample_dec_out == PAD_ID).nonzero()
        
        if len(pad_pos_in) > 0:
            valid_len = pad_pos_in[0].item()
        else:
            valid_len = len(sample_dec_in)
            
        if valid_len > 1:
            if torch.equal(sample_dec_in[1:valid_len], sample_dec_out[:valid_len-1]):
                print("✅ Teacher forcing格式验证通过")
            else:
                print("❌ Teacher forcing格式验证失败")
                print(f"  解码器输入[1:{valid_len}]: {sample_dec_in[1:valid_len]}")
                print(f"  解码器输出[:{valid_len-1}]: {sample_dec_out[:valid_len-1]}")
        
        # 显示样本
        print(f"\n样本示例:")
        for i in range(min(2, enc_data.shape[0])):
            enc_tokens = enc_data[i][enc_data[i] != 0]
            dec_in_tokens = dec_in_data[i][dec_in_data[i] != PAD_ID]
            dec_out_tokens = dec_out_data[i][dec_out_data[i] != PAD_ID]
            
            try:
                en_text = en_tokenizer.decode(enc_tokens.tolist(), skip_special_tokens=True)
                de_in_text = de_tokenizer.decode(dec_in_tokens.tolist(), skip_special_tokens=True)
                de_out_text = de_tokenizer.decode(dec_out_tokens.tolist(), skip_special_tokens=True)
                
                print(f"  样本 {i+1}:")
                print(f"    英文: {en_text}")
                print(f"    德文输入: {de_in_text}")
                print(f"    德文输出: {de_out_text}")
                
            except Exception as e:
                print(f"    解码失败: {e}")
                
    except Exception as e:
        print(f"验证失败: {e}")

print("\n数据预处理完成！")