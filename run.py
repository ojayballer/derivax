from configs.model_config import d_model, d_model_output, vocab_size, N, n
from configs.train_config import batch_size, epochs
from scripts.train import Train

t = Train(d_model, d_model_output, vocab_size, N, n)
t.train(batch_size, epochs)