# derivax

A minimal seq2seq Transformer for symbolic math, built on raw JAX primitives.

No autograd. No frameworks. Every forward pass, every gradient, every weight update is computed by hand.

The model takes a polynomial as input and returns its derivative. It learns the power rule, sum rule, and constant rule purely from examples, treating differentiation as a character-level sequence-to-sequence translation problem.

```
f(x)  = -3x^3+4x^2-4       f(x)  = 5x+4          f(x)  = -5x^3+x^2-2x
f'(x) = -9x^2+8x            f'(x) = 5              f'(x) = -15x^2+2x-2
```

## what is this

This is a ground-up implementation of the Transformer from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). The entire thing, encoder, decoder, multi-head attention, layer norm, feed-forward, positional encoding, AdamW, cross-entropy loss, is written using nothing but `jax.numpy` for the math. There are no calls to automatic differentiation. There is no `loss.backward()`. Every gradient through every layer is derived by hand and implemented manually.

I wanted to understand how a transformer actually works, not at the API level, but at the level of individual matrix multiplications and their partial derivatives flowing backward through the network. So I built one from scratch, trained it to do calculus, and this is the result.

The model has 15.8M parameters. It was trained on 100k synthetic polynomial-derivative pairs for 80 epochs on a P100 GPU , which took about 2 hours. It perfectly differentiates linear expressions and gets most quadratics right. It struggles on cubics, and I discuss why below.

## architecture

```
Input: "3x^2+2x-5"                          Target: "<SOS> 6x+2"
        |                                            |
   [ Embedding ]                               [ Embedding ]
        |                                            |
   [ + Positional Encoding ]                   [ + Positional Encoding ]
        |                                            |
   +-----------------+                         +---------------------+
   |  Encoder (x3)   |                         |   Decoder (x3)      |
   |                  |                         |                     |
   |  Multi-Head      |                         |   Masked Multi-Head |
   |  Attention       |                         |   Attention         |
   |       |          |                         |        |            |
   |  Add & Norm      |                         |   Add & Norm        |
   |       |          |        K, V             |        |            |
   |  Feed-Forward    | ----------------------- |   Cross-Attention   |
   |       |          |                         |        |            |
   |  Add & Norm      |                         |   Add & Norm        |
   +-----------------+                         |        |            |
                                                |   Feed-Forward      |
                                                |        |            |
                                                |   Add & Norm        |
                                                +---------------------+
                                                         |
                                                  [ Linear (d, vocab) ]
                                                         |
                                                    [ Softmax ]
                                                         |
                                                  Output: "6x+2"
```

Config: `d_model=512`, `d_ff=1024`, `heads=8`, `d_k=64`, `layers=3`, `vocab=18`, `params=15.8M`

The architecture is a standard encoder-decoder Transformer following the original paper. Post-norm residual connections, sinusoidal positional encoding, 8-head attention with d_k=64, and a two-layer feed-forward network expanding to 1024 dimensions. Three encoder layers, three decoder layers. The decoder uses causal masking for autoregressive generation and cross-attention to condition on the encoder output. The final layer projects to an 18-token vocabulary and applies softmax.

Everything that touches a weight matrix has a hand-written backward pass. The attention backward computes gradients through the softmax, the scaled dot-product, and all four projection matrices. The layer norm backward handles the full expression with variance and mean terms. The embedding backward accumulates gradients at looked-up indices. None of this is delegated to a framework.

## results

Trained on 100k synthetic examples (polynomials up to degree 3, integer coefficients in the range -5 to 5). Training converged within the first 10 epochs and plateaued around 0.08 cross-entropy loss.

![Training Loss](assets/loss_curve.png)

### accuracy by polynomial degree

| Degree | Accuracy | Example |
|--------|----------|---------|
| 1 (linear) | **100%** | `5x+4` then `5` |
| 2 (quadratic) | **70.3%** | `x^2-5x-1` then `2x-5` |
| 3 (cubic) | **34.1%** | `-5x^3+x^2-2x` then `-15x^2+2x-2` |

### where it fails and why

The model learns the structural rules of differentiation perfectly. It knows that the exponent drops by one, that the old exponent becomes a multiplier, and that constants vanish. What trips it up is the implicit arithmetic. To differentiate `5x^3`, the model needs to output `15x^2`, which means it needs to compute 5 times 3 equals 15 at the character level. It has no calculator. It is doing multiplication by pattern matching over token sequences, and when the numbers get large enough, it starts guessing.

This is not an architecture problem. This is a known limitation of character-level transformers on arithmetic tasks. Greedy decoding makes it worse, because a single wrong digit early in the coefficient cascades through the rest of the sequence with no way to recover. The learned representations are correct. The bottleneck is the decoding strategy.

## project structure

```
derivax/
  model/
    Transformer.py          # encoder-decoder transformer
    Encoder.py              # single encoder layer
    Decoder.py              # single decoder layer
    encoderblock.py         # encoder stack
    decoderblock.py         # decoder stack
    layers/
      MultiHeadAttention.py # scaled dot-product + multi-head attention
      LayerNorm.py          # layer normalization
      FeedForward.py        # two-layer MLP with ReLU
      dense.py              # linear layer (Xavier init)
      Activation.py         # ReLU and Softmax with backward passes
      embedding.py          # token embedding with gradient accumulation
      PositionalEncoding.py # sinusoidal positional encoding
    optim/
      AdamW.py              # AdamW optimizer
      CELoss.py             # cross-entropy loss with padding mask
  data/
    tokenizer.py            # character-level tokenizer (18 tokens)
    datasets.py             # data loading and batching
    generator.py            # synthetic data generation via SymPy
  scripts/
    train.py                # training loop
  configs/
    model_config.py
    train_config.py
  utils/
    checkpoint.py           # model serialization
  evaluate.py               # greedy decoding inference
  run.py                    # entry point
```

## usage

Generate training data:

```bash
python data/generator.py
```

Train:

```bash
python run.py
```

Evaluate:

```bash
python evaluate.py
```

## scope and limitations

The model differentiates flat, univariate polynomials with positive integer exponents up to degree 5 and coefficients between -5 and 5. It handles the power rule, sum rule, and constant rule.

It does not support the product rule, quotient rule, chain rule, trigonometric functions, logarithms, fractions, nested expressions, or variables other than x. These are all out of vocabulary. The tokenizer is character-level and operates on 18 symbols: x, +, -, ^, digits 0 through 9, and three control tokens (PAD, SOS, EOS).

## references

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS*.

Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.

Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.

## license

MIT
