import jax 
import jax.numpy as jnp

class AdamW:
    def __init__(self):
        self.alpha = 0.0001 ;self.B1 = 0.9 ;self.B2 = 0.999 ;self.epsilon = 1e-8 ;self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t+=1

    def update(self,layer_name,weights,output_gradient):
        
        if  layer_name not in self.m:
            self.m[layer_name]=jnp.zeros_like(weights) #momentum
            self.v[layer_name]=jnp.zeros_like(weights) #velocity


        #weight decay 
        weight_decay=weights*(1-(self.alpha*0.01))



        self.m[layer_name]=self.B1*self.m[layer_name]+(1-self.B1)*output_gradient
        
        self.v[layer_name]=(self.B2* self.v[layer_name])+((1-self.B2) * output_gradient**2)
        
         #update first moment estimate
        m_cap=self.m[layer_name]/(1-self.B1**self.t)

        #updatre RMSprop
        v_cap=self.v[layer_name]/(1-self.B2**self.t)

       #update 
        return weight_decay-self.alpha*(m_cap/(jnp.sqrt(v_cap)+self.epsilon))