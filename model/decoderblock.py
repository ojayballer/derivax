from model.Decoder import Decoder
class DecoderBlock:
    def __init__(self,d_model,d_model_output,N,n,adamW): # n is the number of decoder blocks
        self.n=n 
        self.decoder_stack=[Decoder(d_model,d_model_output,N,adamW,id=i) for i in range(self.n)]

    def forward(self,encoder_output,decoder_input,look_ahead_mask,enc_padding_mask):
        for i in range(self.n):
            output=self.decoder_stack[i].forward(encoder_output,decoder_input,look_ahead_mask,enc_padding_mask) #encoder output is project into N dense layers
            decoder_input=output
        return output
    

    def backward(self,output_gradient):
        total_encoder_input_grad=0
        for i in reversed(range(self.n)):
            encoder_input_grad,decoder_input_gradient=self.decoder_stack[i].backward(output_gradient)
            output_gradient=decoder_input_gradient # input gradient from last decoder layer
            total_encoder_input_grad+=encoder_input_grad 
          # i have to sum all the encoder input gradints from each decoder to get the total encoder input gradinet that would flow back to the last encoder layer
        
        return total_encoder_input_grad,decoder_input_gradient 

    
