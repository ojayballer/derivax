from model.layers.dense import Dense
from model.layers.Activation import Softmax as Softmax
import jax.numpy as jnp
from model.layers.PositionalEncoding import PositionalEncoding
class MultiHeadAttention :  ##handles masked multi head attention,ecoder's mha and decoder's mha 
    #-----------------------------------multi head  attention---------------------------------------------------------------------------#####
    
    # self.positionalencoder=PositionalEncoding(embedded_tokens)
       # batch_size,max_seq_len,self.d_model =embedded_tokens
    def __init__(self,d_model,d_model_output,AdamW,N,offset=0): ##n is the number of heads 
        self.N=N
        self.d_model_output=d_model_output
    
        self.AdamW=AdamW
        self.d_model=d_model
       
        self.d_k=self.d_model//N
        self.softmax=Softmax() ###

        self.WQ=Dense(d_model,d_model,self.AdamW,seed=offset)
        self.WK=Dense(d_model,d_model,self.AdamW,seed=offset+1)
        self.WV=Dense(d_model,d_model,self.AdamW,seed=offset+2)
  
        
        self.mh_dense=Dense(d_model,d_model,self.AdamW,seed=offset+3) # for the multi-head attention
     

    
    def forward(self,Q=None,KV=None,mask=None): # pass mask argument  to implement masked multi head attention if needed ,lso for decoder's mha KV output from encoder into decoder's mha +Q from decoder's masked mha
        self.Q=Q ;self.KV=KV 
        
        self.Q_forward_gradient=self.WQ.forward(self.Q) #input*Q_weights +bias   
        self.K_forward_gradient=self.WK.forward(self.KV)# input *K_weights + bias
        self.V_forward_gradient =self.WV.forward(self.KV) #input * \V_weights +bias'''

        # reshape from (batch,seq,d_model) too (batch,seq,N,d_k) and transpose to (btch,N,seq,d_k)
        self.Q_heads=jnp.reshape(self.Q_forward_gradient,(self.Q.shape[0],self.Q.shape[1],self.N,self.d_k)).transpose(0,2,1,3) #(batch,N,seq,d_k)*
        self.K_heads=jnp.reshape(self.K_forward_gradient,((self.KV.shape[0],self.KV.shape[1],self.N,self.d_k))).transpose(0,2,1,3)
        self.V_heads=jnp.reshape(self.V_forward_gradient,(self.KV.shape[0],self.KV.shape[1],self.N,self.d_k)).transpose(0,2,1,3)

        #attention forward pass
        self.scores=self.Q_heads@ self.K_heads.transpose(0,1,3,2)/jnp.sqrt(self.d_k) ##(batch,N,seq,d_k)*(batch,N,d_k,seq)=(batch,N,seq,seq)

        if mask is not None :
            self.scores=self.scores +mask 
        self.attention_weights=self.softmax.forward(self.scores) #(batch,N,seq_len,seq_len)
        Attention = self.attention_weights@ self.V_heads # (batch,N,seq_len,d_k)# attention of all N-heads


        #transpose back to (batch,seq,N,d_k)
        Attention=jnp.transpose(Attention,(0,2,1,3))

        #concatenate all heads by reshaping back to the original shape
        Attention=jnp.reshape(Attention,(self.Q.shape[0],self.Q.shape[1],self.d_model))   #(batch,seq_len,d_model)
        
        #pass multi-head attention head into dense layer
        
        MultiHeadAttention=self.mh_dense.forward(Attention)

        return MultiHeadAttention

        
    

    def backward(self,output_gradient): #output_gradient= dL/dMha

        output_gradient=self.mh_dense.backward(output_gradient)
        
        #reshape output gradient(batch,seq,d_model) to 4D (batch,seq,N,d_k) and  transpose to (batch,N,seq,d_k)
        output_gradient=jnp.reshape(output_gradient,(self.Q.shape[0],self.Q.shape[1],self.N,self.d_k)).transpose(0,2,1,3)
      
         # dL/dV =dA/dV *dL/dA==dA/dV * output_gradient  and reshape to (batch,seq,d_model)
        V_backward_gradient= jnp.transpose(self.attention_weights,(0,1,3,2)) @ output_gradient
        V_backward_gradient=jnp.transpose(V_backward_gradient,(0,2,1,3))
        self.V_backward=jnp.reshape(V_backward_gradient,(self.KV.shape[0],self.KV.shape[1],self.d_model)) #reshape to 3d
        self.V_backward=self.WV.backward(self.V_backward)
     
        #dL/dAw=dA/dAw *dL/dA
        d_attention_weights=  output_gradient @ self.V_heads.transpose(0,1,3,2)  # shpe-->(batch,N,seq,seq)
         
        #dL/d_scores=dL/dAW *dAw/scores
        d_scores=self.softmax.backward(d_attention_weights) #softmax.bw returns d_scores

        #dL/dQ=d_scores *d_scores/dQ  #(batch,N,seq,seq) @(batch,N,seq,d_k) and reshape ack to 3d() -->(batch,seq,d_model)
        Q_backward_gradient =d_scores @ self.K_heads/jnp.sqrt(self.d_k)
        Q_backward_gradient=jnp.transpose(Q_backward_gradient,(0,2,1,3))  # (batch,N,seq,d_k) back to   (batch,seq,N,d_k) 
        self.Q_backward=jnp.reshape(Q_backward_gradient,(self.Q.shape[0],self.Q.shape[1],self.d_model))
        self.Q_backward=self.WQ.backward(self.Q_backward)

        #dL/dK=d_scores * d_scores/dK
        K_backward_gradient=jnp.transpose(d_scores,(0,1,3,2))@ self.Q_heads/jnp.sqrt(self.d_k) 
        K_backward_gradient=jnp.transpose(K_backward_gradient,(0,2,1,3))  #(batch,seq,N,d_k) and then flatten to 3d so N collapses
        self.K_backward=jnp.reshape(K_backward_gradient,(self.KV.shape[0],self.KV.shape[1],self.d_model))
        self.K_backward=self.WK.backward(self.K_backward)

        

        return self.Q_backward ,(self.K_backward+self.V_backward) # Q_backward,KV_backward



