import jax
import jax.numpy as jnp
class LayerNorm:
    def __init__(self,d_model,adamW,seed):
        self.d_model=d_model
        self.epsilon=1e-6
        self.dim=self.d_model ;self.gamma=jnp.ones(self.d_model,) ;self.beta=jnp.zeros(self.d_model,)
        self.adamW=adamW
       
    def forward(self,input):
        self.input=input
        mu=jnp.mean(self.input,axis=-1,keepdims=True) #average over last_dim(d_model) and keep dimension as input dim 
        self.sigma_square=jnp.mean(jnp.square(self.input-mu),axis=-1,keepdims=True) # average over d_model(last dim)
        self.x_bar=(self.input-mu)/jnp.sqrt(self.sigma_square+self.epsilon) 
        return self.gamma * self.x_bar +self.beta
    

    def backward(self,output_gradient):
       #dL/dgamma=dL/dy *dy/dgamma
       d_gamma=jnp.sum(output_gradient * self.x_bar,axis=(0,1))

       #dL/dbeta=dL/dy *dY/dbeta
       d_beta=jnp.sum(output_gradient,axis=(0,1))

        
        #fucking hell,ts is crazy
       input_gradient = (1 / jnp.sqrt(self.sigma_square + self.epsilon)) * ((output_gradient * self.gamma) -
                                                                             (jnp.sum(output_gradient * self.gamma, axis=-1, keepdims=True) / self.d_model) - 
                                                        (self.x_bar / self.d_model) * jnp.sum(output_gradient * self.gamma * self.x_bar, axis=-1, keepdims=True))
       #update gamma
       self.gamma=self.adamW.update(f"{id(self)}_layer_norm_gamma",self.gamma,d_gamma)


       #update beta
       self.beta=self.adamW.update(f"{id(self)}_layer_norm_beta",self.beta,d_beta) #sighs ,finally

       
       

       return input_gradient
    