from model.layers.dense import Dense
from model.layers.Activation import RELU
class FeedForward :
    def __init__(self,input_size,output_size,adamW,offset=0):
        self.dense1=Dense(input_size,output_size,adamW,offset) #exand output size so relu can work using high dimensioanl space ,and then collapse back to d_model after relu
        self.dense2=Dense(output_size,input_size,adamW,offset) #coutput size(model_dim) has to be the input size(model_dim) of the dense1 layer 
        self.relu=RELU()

    def forward(self,input): #ffn is a two layer mLp, dense.fw(act(dense.fw))
        return  self.dense2.forward(self.relu.forward(self.dense1.forward(input))) #max(0,dense.fw(input))-->pass into another dense layer
    
    def  backward(self,output_gradient):
        output_gradient=self.dense2.backward(output_gradient)
        output_gradient=self.relu.backward(output_gradient)
        input_gradient=self.dense1.backward(output_gradient)
        return input_gradient


