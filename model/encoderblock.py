from model.Encoder import Encoder
class EncoderBlock:
    def __init__(self,d_model,d_model_output,N,n,adamW):
          self.n=n ## no of encoders  
          self.encoder_stack=[Encoder(d_model,d_model_output,N,adamW,id=i) for i in range(self.n)]
          

    
    def forward(self,input,padding_mask):  
        
        for i in range(self.n):
             output=self.encoder_stack[i].forward(Q=input,KV=input,mask=padding_mask)
             input=output
        return output
    
    def backward(self,output_gradient):
         for i in reversed(range(self.n)):
              input_gradient=self.encoder_stack[i].backward(output_gradient)
              output_gradient=input_gradient

         return input_gradient
    
              
        