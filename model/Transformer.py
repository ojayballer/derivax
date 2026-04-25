from model.encoderblock import EncoderBlock
from model.decoderblock import DecoderBlock
from model.layers.dense import Dense
from model.layers.Activation import Softmax
from model.optim.AdamW import AdamW
from model.layers.PositionalEncoding import PositionalEncoding
from model.layers.embedding import Embedding
import jax.numpy as jnp
class Transformer :
    def __init__(self,d_model,d_model_output,vocab_size,N,n):
        self.adamw=AdamW()
        self.encoderblock=EncoderBlock(d_model,d_model_output,N,n,self.adamw)
        self.decoderblock=DecoderBlock(d_model,d_model_output,N,n,self.adamw)
        self.dense=Dense(d_model,vocab_size,self.adamw,6)
        self.softmax=Softmax()
        self.encoder_embedding = Embedding(vocab_size, d_model,seed=0, adamw=self.adamw)
        self.decoder_embedding = Embedding(vocab_size, d_model, seed=1,adamw=self.adamw)
        self.positional_encoding = PositionalEncoding(d_model)
    
    def create_mha_mask(self,seq_len):
        mask = jnp.triu(jnp.full((seq_len, seq_len), float('-inf')), k=1)
        return mask

    def create_padding_mask(self, x, pad_id=0):
        mask = (x == pad_id).astype(jnp.float32)
        return jnp.where(mask == 1, float('-inf'), 0.0)[:, None, None, :]
    
    def forward(self, input, decoder_input):
        look_ahead_mask = self.create_mha_mask(decoder_input.shape[1])
        enc_padding_mask = self.create_padding_mask(input)
        dec_padding_mask = self.create_padding_mask(decoder_input)
    
        encoder_input = self.encoder_embedding.forward(input)
        encoder_input = self.positional_encoding.addencodedpositions(encoder_input)
        encoder_output = self.encoderblock.forward(encoder_input, enc_padding_mask)
    
        decoder_input = self.decoder_embedding.forward(decoder_input)
        decoder_input = self.positional_encoding.addencodedpositions(decoder_input)
        decoder_output = self.decoderblock.forward(encoder_output, decoder_input, look_ahead_mask + dec_padding_mask, enc_padding_mask)
    
        Transformer_output = self.dense.forward(decoder_output)
        Transformer_output = self.softmax.forward(Transformer_output)
        return Transformer_output
    
    def backward(self,output_gradient):
        Transformer_output_gradient=self.softmax.backward(output_gradient)
        Transformer_output_gradient=self.dense.backward(Transformer_output_gradient)
        total_encoder_input_grad,decoder_input_gradient=self.decoderblock.backward(Transformer_output_gradient)
        input_gradient_enc=self.encoderblock.backward(total_encoder_input_grad)
        enc_embedding_grad=self.encoder_embedding.backward(input_gradient_enc)
        decoder_embedding_grad=self.decoder_embedding.backward(decoder_input_gradient)
        return None # sinusodial positional encoding 

        
        
