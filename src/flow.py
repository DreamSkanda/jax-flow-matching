import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.experimental import ode
from jax.scipy.stats import norm
from functools import partial

def NeuralODE(vec_field_net, dim):

    def divergence_fwd(f):
        def _div_f(params, x, t):
            jac = jax.jacfwd(lambda x: f(params, x, t))
            return jnp.trace(jac(x))
        return _div_f

    def base_logp(x):
        return norm.logpdf(x).sum(-1)
    
    @partial(vmap, in_axes=(None, 0), out_axes=(0,0))
    def forward(params, x0):
        def _ode(state, t):
            x = state[0]  
            return vec_field_net(params, x, t), \
                - divergence_fwd(vec_field_net)(params, x, t)
        
        logp0 = base_logp(x0)

        xt, logpt = ode.odeint(_ode,
                 [x0, logp0],
                 jnp.array([0.0, 1.0]),
                 rtol=1e-10, atol=1e-10,
                 mxstep=20000
                 )
        return xt[-1], logpt[-1]

    @partial(vmap, in_axes=(None, 0), out_axes=(0,0))
    def reverse(params, xt):
        def _ode(state, t):
            x = state[0]     
            return - vec_field_net(params, x, -t), \
                divergence_fwd(vec_field_net)(params, x, -t)
        
        logpt = 0.0
        
        x0, logp0 = ode.odeint(_ode,
                 [xt, logpt],
                 jnp.array([-1.0, 0.0]),
                 rtol=1e-10, atol=1e-10,
                 mxstep=20000
                 )
        return x0[-1], base_logp(x0[-1]) - logp0[-1]

    def batched_sample_fun(rng, params, sample_size):
        x0 = random.normal(rng, (sample_size, dim))

        return forward(params, x0)

    @partial(vmap, in_axes=(None, 0), out_axes=0)
    def logp_fun(params, xt):
        def _ode(state, t):
            x = state[0]     
            return - vec_field_net(params, x, -t), \
                divergence_fwd(vec_field_net)(params, x, -t)
        
        logpt = 0.0

        x0, logp0 = ode.odeint(_ode,
                 [xt, logpt],
                 jnp.array([-1.0, 0.0]),
                 rtol=1e-10, atol=1e-10,
                 mxstep=20000
                 )
        return base_logp(x0[-1]) - logp0[-1]
    
    return forward, reverse, batched_sample_fun, logp_fun

if __name__ == '__main__':
    from jax.config import config
    config.update("jax_enable_x64", True)

    from jax.example_libraries.stax import serial, Dense, Relu
    from jax.nn.initializers import zeros

    n = 2
    dim = 2
    sample_size = 10

    def make_vec_field_net(rng):
        net_init, net_apply = serial(Dense(512), Relu, Dense(512), Relu, Dense(n*dim))
        in_shape = (-1, n*dim+1)
        _, net_params = net_init(rng, in_shape)

        def net_with_t(params, x, t):
            return net_apply(params, jnp.concatenate((x,t.reshape(1))))
        
        return net_params, net_with_t

    init_rng, rng = random.split(random.PRNGKey(42))
    params, vec_field_net = make_vec_field_net(init_rng)

    forward, reverse, batched_sampler, logp_fun = NeuralODE(vec_field_net, n*dim)

    x0 = random.normal(random.PRNGKey(42), (sample_size, n*dim))

    xt, logpt = forward(params, x0)
    x0r, logptr = reverse(params, xt)
    print((logpt - logptr) / logpt)
    print((x0 - x0r) / x0)






