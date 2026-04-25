import jax 
import jax.numpy as jnp
class Embedding:
    def __init__(self,vocab_size,d_model,seed,adamw):
        self.d_model=d_model
        key = jax.random.PRNGKey(seed)
        self.weights = jax.random.normal(key, (vocab_size, self.d_model)) * 0.01
        self.adamW = adamw

    def forward(self,input): ##(batch_size,seq_len)
        self.input=input
        return self.weights[input] * jnp.sqrt(self.d_model) ## project each seq_len[i] to a d_model vector -->(batch_size,seq_len,d_model)
    
    def backward(self,output_gradient):  ## (batch,seq_len,d_model)
        d_weights = jnp.zeros_like(self.weights)  ##temporary gradient accumulator 
        d_weights = d_weights.at[self.input].add(output_gradient)
        self.weights=self.adamW.update(f"{id(self)}_weights",self.weights,d_weights)
        return None 
