import jax
import jax.numpy as jnp
class PositionalEncoding :
    def __init__(self,d_model):
        #(batch,seq_len,d_model) embedded tokens
        self.d_model=d_model
        self.positional_embedding=jnp.full((5000,self.d_model),0) #where 5000 is the max_seq_len(max no of tokens in an input)


        i=jnp.arange(0,self.d_model//2)
        div=10000**(2*i/self.d_model) ;div=div.reshape(1,-1)
        pos=jnp.arange(self.positional_embedding.shape[0]) ;pos=pos.reshape(-1,1)
        angles=pos / div

        #sine-pos
        self.positional_embedding=self.positional_embedding.at[:,0::2].set(jnp.sin(angles))  #for  token-i apply sine pos to even  indexes

        #cos-pos
        self.positional_embedding=self.positional_embedding.at[:,1::2].set(jnp.cos(angles))  #for token-i+1 apply cos pos to odd indexes
    

    def addencodedpositions(self, embedded_tokens):
        seq_len = embedded_tokens.shape[1]
        return self.positional_embedding[None, :seq_len, :] + embedded_tokens