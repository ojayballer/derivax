from model.Transformer import Transformer
from data.datasets import Datasets
from model.optim.CELoss import CategoricalCrossEntropy
import time
from utils.checkpoint import save,save_losses
class Train :
    def __init__(self,d_model,d_model_output,vocab_size,N,n):
        self.transformer=Transformer(d_model,d_model_output,vocab_size,N,n)
        self.datasets=Datasets()
        self.loss=CategoricalCrossEntropy()

    def train(self,batch,epochs):
        start=time.time()
        self.datasets.load_data()
        losses=[] 
        for i in range(epochs):
           print(f"Starting Training for Epoch {i+1}/{epochs}")
           batches=self.datasets.batching(batch_size=batch)
           for b,(enc_data_batch, dec_data_batch, target_data_batch) in enumerate(batches):
                output = self.transformer.forward(enc_data_batch, dec_data_batch)
                loss=self.loss.forward(output,target_data_batch) ;losses.append(float(loss))
                output_gradient=self.loss.backward(output,target_data_batch)

                 #step adamw
                self.transformer.adamw.step()
                
                ### update weights 
                self.transformer.backward(output_gradient)
                #if (b + 1) % 100 == 0:
                print(f"Epoch{i+1}/{epochs},Batch: {b+1}/{len(batches)},Loss :{loss},Time: {time.time() - start:.2f}s")
           save_losses(losses) ##save losses from ach epoch
           if(i+1)==epochs:
               self.transformer.adamw.m = {}
               self.transformer.adamw.v = {}
               save(self.transformer, i+1)


        return losses