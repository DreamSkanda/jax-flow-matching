import jax.numpy as jnp
from jax import vmap
from functools import partial

def make_loss(vec_field_net):

    @partial(vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def _matching(params, x0, x1, t):
        x = t*x1 + (1 - t)*x0
        return jnp.linalg.norm((x1 - x0) - vec_field_net(params, x, t))

    def loss(params, x0, x1, t):
        m = _matching(params, x0, x1, t)
        m_mean = jnp.mean(m)
        m_std = jnp.std(m) / jnp.sqrt(m.shape[0])
        return m_mean, m_std

    return loss

if __name__ == '__main__':
    from jax.example_libraries.stax import serial, Dense, Relu
    from jax.nn.initializers import zeros
    from flow import make_cond_flow
    from jax import random

    n = 2
    dim = 2
    sample_size = 10
    sigma_min = 0.01

    def make_vec_field_net(rng):
        net_init, net_apply = serial(Dense(512), Relu, Dense(512), Relu, Dense(n*dim, W_init=zeros, b_init=zeros))
        in_shape = (-1, n*dim+1)
        _, net_params = net_init(rng, in_shape)

        def net_with_t(params, x, t):
            return net_apply(params, jnp.concatenate((x,t.reshape(1))))
        
        return net_params, net_with_t

    init_rng, rng_1, rng_2 = random.split(random.PRNGKey(42), num=3)
    params, vec_field_net = make_vec_field_net(init_rng)
    _, cond_vec_field = make_cond_flow(sigma_min)

    loss = make_loss(vec_field_net, cond_vec_field)

    print(loss(params, random.normal(rng_1, (sample_size, n*dim)), jnp.ones((sample_size, n*dim)), jnp.zeros((sample_size,))))