from data.tokenizer import Tokenizer
import jax.numpy as jnp
import random
class Datasets :
    def __init__(self):
        self.tokenize=Tokenizer()

    def load_data(self):
        with open('data/train/inputs.txt') as f :
            inputs=f.readlines()
        ## tokenize inputs 
        encoder_inputs=[self.tokenize.encode(inputs[i].strip()) for i in range(len(inputs))]


        with open('data/train/outputs.txt') as f:
            outputs=f.readlines()

        ##tokenize outputs
        decoder_input=[[self.tokenize.SOS] + self.tokenize.encode(outputs[i].strip())  for i in range(len(outputs))]
        target=[self.tokenize.encode(outputs[i].strip()) + [self.tokenize.EOS] for i in range(len(outputs))]
        

        ## pad all 
        max_enc_len = max(len(s) for s in encoder_inputs)  
        max_dec_len = max(len(s) for s in decoder_input)

        self.encoder_inputs=[s+[self.tokenize.PAD] * (max_enc_len - len(s)) for s in encoder_inputs]
        self.decoder_input=[s+[self.tokenize.PAD]*(max_dec_len - len(s)) for s in decoder_input]
        self.target=[s+[self.tokenize.PAD] * (max_dec_len - len(s)) for s in target]

   

    def batching(self, batch_size):
       indices = list(range(len(self.encoder_inputs)))
       random.shuffle(indices)
       batches = []
       for i in range(0, len(indices), batch_size):
             batch_idx = indices[i:i+batch_size]
             encoder_batch = jnp.array([self.encoder_inputs[j] for j in batch_idx])
             decoder_batch = jnp.array([self.decoder_input[j] for j in batch_idx])
             target_batch = jnp.array([self.target[j] for j in batch_idx])
             batches.append((encoder_batch, decoder_batch, target_batch))

       return batches

        
