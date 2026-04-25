from utils.checkpoint import load
from data.tokenizer import Tokenizer
import jax.numpy as jnp

transformer = load(epoch=80)
tokenizer = Tokenizer()

def greedy_decode(transformer, tokenizer, input_str, max_len=50):
    enc_input = jnp.array([tokenizer.encode(input_str)])
    dec_input = jnp.array([[tokenizer.SOS]])

    for _ in range(max_len):
        output = transformer.forward(enc_input, dec_input)
        next_token = jnp.argmax(output[0, -1, :])
        if next_token == tokenizer.EOS:
            break
        dec_input = jnp.concatenate([dec_input, next_token.reshape(1, 1)], axis=1)

    tokens = dec_input[0, 1:].tolist()
    return tokenizer.decode(tokens)

test_cases = [
    "3x^2+2x-5",
    "-4x^3+x^2-3x+1",
    "5x-3",
    "x^3-2x^2+x",
    "-x^2+4x-1",
]

for inp in test_cases:
    print(f"f(x)  = {inp}")
    print(f"f'(x) = {greedy_decode(transformer, tokenizer, inp)}")
    print()