import jax.numpy as jnp
class CategoricalCrossEntropy:

    def forward(self, y_pred, y_true, pad_id=0):
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -jnp.log(y_pred[jnp.arange(y_pred.shape[0])[:, None],
                                jnp.arange(y_pred.shape[1])[None, :],
                                y_true])
        mask = (y_true != pad_id).astype(jnp.float32)
        loss = loss * mask
        return jnp.sum(loss) / jnp.sum(mask)

    def backward(self, y_pred, y_true, pad_id=0):
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        batch, seq_len, vocab_size = y_pred.shape
        mask = (y_true != pad_id).astype(jnp.float32)
        output_gradient = jnp.zeros_like(y_pred)

        output_gradient = output_gradient.at[jnp.arange(batch)[:, None],
                                             jnp.arange(seq_len)[None, :],
                                             y_true].set(-1 / y_pred[jnp.arange(batch)[:, None],
                                                                      jnp.arange(seq_len)[None, :],
                                                                      y_true])
        output_gradient = output_gradient * mask[:, :, None]
        
        return output_gradient / jnp.sum(mask) ##scale output gradient