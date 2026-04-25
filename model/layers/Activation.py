import jax.numpy as jnp
class RELU :
    def __init__(self):
        pass
        

    def forward(self,input):
        self.input=input 
        return jnp.maximum(0,self.input)
    
    def backward(self,output_gradient):
        return output_gradient *(self.input >0 )


class Softmax :
    def __init__(self):
        pass

    def forward(self,input):
        self.input =input #linear layer into softmax ->(batch,seq,d_model) /(batch,N,seq_len,seq_len) ->mh-attention into softmax ///same otput shape too
        input_max = jnp.max(self.input, axis=-1, keepdims=True)
        exps = jnp.exp(self.input-input_max)
        sum_exps = jnp.sum(exps, axis=-1, keepdims=True)
    
        # Return the output
        self.output = jnp.where(sum_exps == 0, 0.0, exps / sum_exps)
        return self.output
    
    def backward(self,output_gradient) :
        sum_output_gradient=jnp.sum(output_gradient*self.output,axis=-1,keepdims=True)
        return self.output *( output_gradient -sum_output_gradient)


        