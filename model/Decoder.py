from model.layers.MultiHeadAttention import MultiHeadAttention 
from model.layers.LayerNorm import LayerNorm
from model.layers.FeedForward import FeedForward
from model.optim.AdamW  import AdamW
from model.layers.dense import Dense
class Decoder :
    def __init__(self,d_model,d_model_output,N,adamW,id=0):
        offset = id * 10 + 100
        #create mask
        self.adamW=adamW
        self.masked_mha=MultiHeadAttention(d_model,d_model_output,self.adamW,N,offset=offset)
        self.mha=MultiHeadAttention(d_model,d_model_output,self.adamW,N,offset=offset+4)
        self.norm1=LayerNorm(d_model,self.adamW,offset+8)
        self.norm2=LayerNorm(d_model,self.adamW,offset+9)
        self.norm3=LayerNorm(d_model,self.adamW,offset+10)
        self.FFN=FeedForward(d_model,d_model_output,self.adamW,offset=offset+11)
        
    def forward(self,encoder_output,decoder_input,look_ahead_mask,enc_padding_mask): #output embedding shifted right,#encoder's last output
        masked_mha=self.masked_mha.forward(Q=decoder_input,KV=decoder_input,mask=look_ahead_mask)
        masked_mha=decoder_input+masked_mha 
        Q=self.norm1.forward(masked_mha)

        #____________________multi-head-attention___________________________#
    
        mha=self.mha.forward(Q,KV=encoder_output,mask=enc_padding_mask)
        mha=Q+mha 
        mha=self.norm2.forward(mha)
        ffn=self.FFN.forward(mha)
        ffn=ffn+mha
        ffn=self.norm3.forward(ffn)
            
        return ffn #next decoder block takes this as input 
     
    
    def backward(self,output_gradient):
        ffn_total_grad=self.norm3.backward(output_gradient)
        ffn=self.FFN.backward(ffn_total_grad)
        mha_grad=self.norm2.backward(ffn_total_grad+ffn)
        Q_grad=mha_grad
        Q,encoder_input_grad=self.mha.backward(mha_grad)
        Q_final=self.norm1.backward(Q+Q_grad)
        Q_back,KV_back=self.masked_mha.backward(Q_final)
        decoder_input_grad=Q_back+KV_back+Q_final

        return encoder_input_grad ,decoder_input_grad



        