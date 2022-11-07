import jax.numpy as jnp
from jax import vmap
from functools import partial

def make_loss(vec_field_net, cond_vec_field):

    @partial(vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def _matching(params, x, x1, t):
        return jnp.linalg.norm(vec_field_net(params, x, t) - cond_vec_field(x, x1, t))

    def loss(params, x, x1, t):
        m = _matching(params, x, x1, t)
        m_mean = jnp.mean(m)
        m_std = jnp.std(m) / jnp.sqrt(m.shape[0])
        return m_mean, m_std

    return loss