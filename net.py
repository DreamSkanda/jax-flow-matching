import jax.numpy as jnp
from jax.example_libraries.stax import serial, Dense, Relu
from jax.nn.initializers import zeros

def make_vec_field_net(rng, dim):
    net_init, net_apply = serial(Dense(512), Relu, Dense(512), Relu, Dense(dim, W_init=zeros, b_init=zeros))
    in_shape = (-1, dim+1)
    _, net_params = net_init(rng, in_shape)

    def net_with_t(params, x, t):
        return net_apply(params, jnp.concatenate((x,t.reshape(1))))
    
    return net_params, net_with_t