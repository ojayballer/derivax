from model.layers.embedding import Embedding 
from model.layers.PositionalEncoding import PositionalEncoding
from model.layers.MultiHeadAttention import MultiHeadAttention 
from model.optim.AdamW import AdamW
from model.layers.Activation import Softmax,RELU
from model.layers.FeedForward import FeedForward
from model.layers.LayerNorm import LayerNorm

 #self.embed_token=TokenEmbedding(input)
 ## work in progress -->  
##self.encode_pos=PositionalEncoding(self.embed_token.)
#final input embedding
#self.final_input=self.encode_pos.addencodedpositions()

class Encoder :  # a single Encoder
    def __init__(self,d_model,d_model_output,N,adamW,id=0): #N is the number of heads I will use for multi-head attention
        offset = id * 10
        self.adamW=adamW
        
        self.softmax=Softmax() 
        #mha
        self.mha=MultiHeadAttention(d_model,d_model_output,self.adamW,N,offset=offset)
        self.norm1=LayerNorm(d_model,self.adamW,offset+4)
        self.norm2=LayerNorm(d_model,self.adamW,offset+5)
        self.FFN=FeedForward(d_model,d_model_output,self.adamW,offset=offset+6) ### 


    def forward(self,Q=None,KV=None,mask=None):  #for encoder Q and KV are equal
        mha=self.mha.forward(Q,KV,mask=mask)
        mha=Q+mha  ;mha=self.norm1.forward(mha)
        FFn=self.FFN.forward(mha) ; output=mha+FFn ; output= self.norm2.forward(output) # add and norm output 
        return output  ## if it is the last encoder layer, output would be projected into K and V by  going into two  dense layers and then into the decoder's mha else output flows into the next encoder layer
    
    def backward(self,output_gradient): # the merging of K and V of the last encoder would be handled in the encoder block,but for now we assume K and V are merged already
        output_gradient=self.norm2.backward(output_gradient) 
         #split output gradient into mha_grda and ffn_grad
        FFn_grad=self.FFN.backward(output_gradient)
        mha_grad=self.norm1.backward(output_gradient+FFn_grad)

        Q_grad=mha_grad ; Q_backward,KV_backward=self.mha.backward(mha_grad)

        
        
        ##input gradient
        # this would go to the previous encoder layer and flow back to positional encoder (but it is not used since I used sinusodial pos enc does not treat positon as trainable parameters)
        return Q_grad+Q_backward +KV_backward ## input_gradients 

        

        