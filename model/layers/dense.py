import jax 
import jax.numpy as jnp
class Dense :
    def __init__(self,input_size,output_size,AdamW,seed) : #input shape=putput_shape==(batch_size,max_seq_len,input_size/output_size)
         
          nk,ok=jax.random.split(jax.random.PRNGKey(seed))
     

          #xavier initaialisation
          std=jnp.sqrt(2/(input_size+output_size))

          self.weights=jax.random.normal(nk,(input_size,output_size))*std

          self.bias=jnp.zeros(output_size)   
          self.AdamW=AdamW
           

    def forward(self,input ):
         self.input=input
         return  input @ self.weights + self.bias 
    
    
    def backward(self,output_gradient):
         #dl/dw=dl/output *douptu/dw=output_gradient*input
         weights_gradient=jnp.sum(jnp.transpose(self.input,axes=(0,2,1)) @ output_gradient,axis=0) #project to 2d by summing over batch  ## (batch_size,input_size,max_seq_len)*(batch,_maxseq_len,output_size) ==(input_size,output_size)
         
         #dl/di=doutput/di *dl/doutput=weights*utput_gradient
         input_gradient=jnp.matmul(output_gradient,self.weights.T)# (batch,seq,output_size) *(output_size,input_size)

         #dl/db=dl/output*doutput/db=dl/doutput ,sum over the batch and seq to get (output,)
         bias_gradient=jnp.sum(output_gradient,axis=(0,1))

         self.weights=self.AdamW.update(f"{id(self)}_weights",self.weights,weights_gradient)
         #updte bias
         self.bias=self.AdamW.update(f"{id(self)}_bias",self.bias,bias_gradient)

         return input_gradient