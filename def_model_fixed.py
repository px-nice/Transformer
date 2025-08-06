import torch
from torch import nn
from torch.utils import data
import numpy as np
import math
from torch import optim
import pickle


def get_vocab_size(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)  # 高效计算行数

src_vocab_size = get_vocab_size('Transformer/val_data/en_tokenizer/vocab.txt')
tgt_vocab_size = get_vocab_size('Transformer/val_data/de_tokenizer/vocab.txt')

#根据论文的base模型的模型超参数
d_model=512
d_ff=2048
d_k=d_v=64
n_heads=8
n_layers=6
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Positional_Encoding(nn.Module):
    '''
    该层的输入为embedding层的输出，输出为位置编码加上输入的结果
    输入输出的形状均为 (batch_size, seq_len, d_model)
    '''
    def __init__(self,d_model,dropout=0.1,max_len=256):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)                      #shape:(max_len,1)
        div_term=torch.exp(-torch.arange(0,d_model,step=2).float()*(math.log(10000)/d_model))  #shape:(d_model//2)
        #position*div_term的shape: (max_len,d_model//2)

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        #pe shape:(max_len,1,d_model),便于后续对batch_size广播
        pe=pe.unsqueeze(0)  # 修复：改为(1, max_len, d_model)

        #训练过程中缓冲区不会更新（不会计算梯度）
        self.register_buffer('pe',pe)
    
    def forward(self,X):
        '''
        X shape:(batch_size, seq_len, d_model)
        '''
        # 修复：正确的维度处理
        X = X + self.pe[:, :X.shape[1], :]
        return self.dropout(X)
    

def get_attn_pad_mask(seq_q,seq_k):
    '''
    seq_q和seq_k分别是Query序列和Key序列
    形状为(batch_size,seq_len)
    '''
    batch_size,len_q=seq_q.shape
    batch_size,len_k=seq_k.shape
    pad_attn_mask=seq_k.data.eq(0).unsqueeze(1) #shape:(batch_size,1,len_k)
    return pad_attn_mask.expand(batch_size,len_q,len_k)

def get_attn_subsequence_mask(seq):
    '''
    seq:(batch_size,tgt_len)
    '''
    attn_shape=[seq.shape[0],seq.shape[1],seq.shape[1]]
    subsequence_mask=np.triu(np.ones(attn_shape),k=1)
    subsequence_mask=torch.from_numpy(subsequence_mask).bool()  # 修复：使用.bool()替代.byte()
    return subsequence_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,Q,K,V,attn_mask):
        '''
        Q:(batch_size,n_heads,len_q,d_k)
        K:(batch_size,n_heads,len_k,d_k)
        V:(batch_size,n_heads,len_V(=len_k)),d_v)
        attn_mask:(batch_size,n_heads,seq_len,seq_len)
        '''
        scores=torch.matmul(Q,K.transpose(-2,-1))/np.sqrt(d_k)  #scores:(batch_size,n_heads,len_q,len_k)
        
        # 修复：添加数值稳定性检查
        scores = torch.clamp(scores, min=-1e9, max=1e10)
        scores.masked_fill_(attn_mask,-1e9) #将布尔矩阵中True的地方填充为-1e9

        attn=nn.Softmax(dim=-1)(scores)
        
        # 修复：添加attention权重的数值稳定性检查
        attn = torch.clamp(attn, min=1e-8, max=1.0)
        
        result=torch.matmul(attn,V) #result:[batch_size,n_heads,len_q,d_v]
        return result,attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_K=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_V=nn.Linear(d_model,d_v*n_heads,bias=False)
        self.fc=nn.Linear(d_v*n_heads,d_model,bias=False)
        self.ln=nn.LayerNorm(d_model)
        
        # 修复：添加权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.W_Q, self.W_K, self.W_V, self.fc]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self,input_Q,input_K,input_V,attn_mask):
        '''
        input_Q:(batch_size,len_q,d_model)
        input_K:(batch_size,len_k,d_model)
        input_V:(batch_size,len_v,d_model)
        attn_mask:(batch_size,len_q,len_k)
        '''
        residual,batch_size=input_Q,input_Q.shape[0]
        Q=self.W_Q(input_Q).view(batch_size,-1,n_heads,d_k).transpose(1,2)  # 修复：调整维度变换顺序
        K=self.W_K(input_K).view(batch_size,-1,n_heads,d_k).transpose(1,2)
        V=self.W_V(input_V).view(batch_size,-1,n_heads,d_v).transpose(1,2)

        attn_mask=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)  #attn_mask:(batch_size,n_heads,len_q,len_k)
        result,attn=ScaledDotProductAttention()(Q,K,V,attn_mask) #result:(batch_size,n_heads,len_q,d_v)  ,attn:(batch_size,n_heads,len_q,len_k)
        result=result.transpose(1,2).reshape(batch_size,-1,n_heads*d_v)#result:(batch_size,len_q,n_heads*d_v)
        output=self.fc(result)  #output:(batch_size,len_q,d_model)
        return self.ln(output+residual),attn
    
class PosWiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),  # 修复：添加dropout
            nn.Linear(d_ff,d_model,bias=False)
        )
        self.ln=nn.LayerNorm(d_model)
        
        # 修复：添加权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self,inputs):
        '''
        inputs: (batch_size,seq_len,d_model)
        '''
        residual=inputs
        output=self.fc(inputs)
        return self.ln(output+residual) #return: (batch_size,seq_len,d_model)
    
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn=MultiHeadAttention()
        self.pos_ffn=PosWiseFeedForward()
    
    def forward(self,enc_inputs,enc_self_attn_mask):
        '''
        enc_inputs:(batch_size,src_len,d_model)
        enc_self_attn_mask(batch_size,len_q,len_k)
        '''
        enc_outputs,attn=self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs=self.pos_ffn(enc_outputs)
        return enc_outputs,attn
    
class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn=MultiHeadAttention()
        self.dec_enc_attn=MultiHeadAttention()
        self.pos_ffn=PosWiseFeedForward()

    def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        '''
        dec_outputs:(batch_size,tgt_len,d_model)
        enc_outputs:(batch_size,src_len,d_model)
        dec_self_attn_mask:(batch_size,tgt_len,tgt_len)
        dec_ecc_attn_mask:(batch_size,tgt_len,src_len)
        '''
        dec_outputs,dec_self_attn=self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        dec_outputs,dec_enc_attn=self.dec_enc_attn(dec_outputs,enc_outputs,enc_outputs,dec_enc_attn_mask)
        dec_outputs=self.pos_ffn(dec_outputs)

        return dec_outputs,dec_self_attn,dec_enc_attn
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_ebd=nn.Embedding(src_vocab_size,d_model,padding_idx=0)
        self.pos_ecd=Positional_Encoding(d_model)
        self.layers=nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        
        # 修复：添加embedding权重初始化
        nn.init.normal_(self.src_ebd.weight, mean=0, std=d_model**-0.5)
        self.src_ebd.weight.data[0].fill_(0)  # padding token的embedding设为0
    
    def forward(self,enc_inputs):
        '''
        enc_inputs:(batch_size,src_len)
        '''
        enc_outputs=self.src_ebd(enc_inputs) #(batch_size,src_len,d_model)
        # 修复：embedding缩放
        enc_outputs = enc_outputs * math.sqrt(d_model)
        enc_outputs=self.pos_ecd(enc_outputs)  # 修复：不需要transpose
        enc_self_attn_mask=get_attn_pad_mask(enc_inputs,enc_inputs)
        enc_self_attns=[]
        for layer in self.layers:
            #enc_outputs: [batch_size,src_len,d_model]  enc_self_attn: [batch_size,n_heads,src_len,src_len]
            enc_outputs,enc_self_attn=layer(enc_outputs,enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs,enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_ebd=nn.Embedding(tgt_vocab_size,d_model,padding_idx=0)
        self.pos_ecd=Positional_Encoding(d_model)
        self.layers=nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        
        # 修复：添加embedding权重初始化
        nn.init.normal_(self.tgt_ebd.weight, mean=0, std=d_model**-0.5)
        self.tgt_ebd.weight.data[0].fill_(0)  # padding token的embedding设为0
    
    def forward(self,dec_inputs,enc_inputs,enc_outputs):
        '''
        dec_inputs:(batch_size,tgt_len)
        enc_inputs:(batch_size,src_len)
        enc_outputs:(batch_size,src_len,d_model)
        '''

        dec_outputs=self.tgt_ebd(dec_inputs) #[batch_size,tgt_len,d_model]
        # 修复：embedding缩放
        dec_outputs = dec_outputs * math.sqrt(d_model)
        dec_outputs=self.pos_ecd(dec_outputs) # 修复：不需要transpose
        dec_self_attn_pad_mask=get_attn_pad_mask(dec_inputs,dec_inputs).to(device)
        dec_self_attn_subsequence_mask=get_attn_subsequence_mask(dec_inputs).to(device) #[batch_size,tgt_len,tgt_len]
        dec_self_attn_mask=torch.gt(dec_self_attn_pad_mask+dec_self_attn_subsequence_mask,0)
        dec_enc_attn_mask=get_attn_pad_mask(dec_inputs,enc_inputs) #[batch_size,tgt_len,src_len]

        dec_self_attns,dec_enc_attns=[],[]
        for layer in self.layers:
            dec_outputs,dec_self_attn,dec_enc_attn=layer(dec_outputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs,dec_self_attns,dec_enc_attns
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()
        self.projection=nn.Linear(d_model,tgt_vocab_size,bias=False)
        self.device=device
        self.to(device=self.device)
        self.d_model=d_model
        
        # 修复：添加输出层权重初始化
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self,enc_inputs,dec_inputs):
        '''
        enc_inputs:[batch_size,src_len]
        dec_inputs:[batch_size,tgt_len]
        '''
        enc_outputs,enc_self_attns=self.encoder(enc_inputs)
        dec_outputs,dec_self_attns,dec_enc_attns=self.decoder(dec_inputs,enc_inputs,enc_outputs)
        dec_logits=self.projection(dec_outputs) #[batch_size,tgt_len,tgt_vocab_size]
        
        # 修复：返回正确的形状，不要reshape
        return dec_logits,enc_self_attns,dec_self_attns,dec_enc_attns

if __name__=='__main__':
    # 假设我们有以下测试参数
    batch_size = 2
    src_len = 10  # 源序列长度
    tgt_len = 8   # 目标序列长度

    # 创建测试输入 - 前面是真实token(值>0)，后面是padding(值为0)
    # 英语输入 (encoder输入)
    enc_inputs = torch.tensor([
        [3, 5, 7, 2, 0, 0, 0, 0, 0, 0],  # 第一个样本：4个真实token + 6个padding
        [4, 9, 1, 6, 8, 2, 0, 0, 0, 0]   # 第二个样本：6个真实token + 4个padding
    ]).to(device)

    # 德语输入 (decoder输入)
    dec_inputs = torch.tensor([
        [5, 3, 8, 1, 0, 0, 0, 0],        # 第一个样本：4个真实token + 4个padding
        [2, 7, 4, 9, 6, 0, 0, 0]         # 第二个样本：5个真实token + 3个padding
    ]).to(device)

    print("测试encoder self-attention mask:")
    enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs).to(device)
    print("Shape:", enc_self_attn_mask.shape)
    print("Mask values:\n", enc_self_attn_mask)

    print("\n测试decoder self-attention pad mask:")
    dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
    print("Shape:", dec_self_attn_pad_mask.shape)
    print("Mask values:\n", dec_self_attn_pad_mask)

    print("\n测试decoder self-attention subsequence mask:")
    dec_self_attn_sub_mask = get_attn_subsequence_mask(dec_inputs).to(device)
    print("Shape:", dec_self_attn_sub_mask.shape)
    print("Mask values:\n", dec_self_attn_sub_mask)

    print("\n测试decoder self-attention combined mask:")
    combined_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_sub_mask, 0).to(device)
    print("Shape:", combined_mask.shape)
    print("Mask values:\n", combined_mask)

    print("\n测试decoder-encoder attention mask:")
    dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs).to(device)
    print("Shape:", dec_enc_attn_mask.shape)
    print("Mask values:\n", dec_enc_attn_mask)