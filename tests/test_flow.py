from config import *
from net import make_vec_field_net
from flow import NeuralODE

def test_logp():

    n = 6
    dim = 2
    batchsize = 10

    key = jax.random.PRNGKey(42)

    params, vec_field_net = make_vec_field_net(key, n, dim)
    _, _, batched_sampler, logp_fun = NeuralODE(vec_field_net, n*dim)

    key, subkey = jax.random.split(key)
    x, logp = batched_sampler(subkey, params, batchsize)
    assert (x.shape == (batchsize, n*dim))
    assert (logp.shape == (batchsize, ))

    logp_inference = logp_fun(params, x)
    
    assert jnp.allclose(logp, logp_inference) 