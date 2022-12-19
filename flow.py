import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.experimental import ode
from jax.scipy.stats import norm
from functools import partial

def make_cond_flow(sigma_min):

    def mu(x1, t):
        return t*x1

    def sigma(x1, t):
        return 1 - (1 - sigma_min)*t

    @partial(vmap, in_axes=(0, 0, 0), out_axes=0)
    def cond_flow(x0, x1, t):
        return sigma(x1, t) * x0 + mu(x1, t)

    def cond_vec_field(x, x1, t):
        return (x1 - (1 - sigma_min)*x)/(1 - (1 - sigma_min)*t)

    return cond_flow, cond_vec_field


def NeuralODE(vec_field_net, dim):

    def divergence_fwd(f):
        def _div_f(params, x, t):
            jac = jax.jacfwd(lambda x: f(params, x, t))
            return jnp.trace(jac(x))
        return _div_f

    def base_logp(x):
        return norm.logpdf(x).sum(-1)
    
    @partial(vmap, in_axes=(None, 0), out_axes=0)
    def forward(params, x0):
        def _ode(x, t):    
            return vec_field_net(params, x, t)
        
        xt = ode.odeint(_ode,
                 x0,
                 jnp.array([0.0, 1.0]),
                 rtol=1e-10, atol=1e-10,
                 mxstep=5000
                 )
        return xt[-1]

    
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
    from jax.example_libraries.stax import serial, Dense, Relu
    from jax.nn.initializers import zeros

    n = 2
    dim = 2
    sample_size = 10

    def make_vec_field_net(rng):
        net_init, net_apply = serial(Dense(512), Relu, Dense(512), Relu, Dense(n*dim, W_init=zeros, b_init=zeros))
        in_shape = (-1, n*dim+1)
        _, net_params = net_init(rng, in_shape)

        def net_with_t(params, x, t):
            return net_apply(params, jnp.concatenate((x,t.reshape(1))))
        
        return net_params, net_with_t

    init_rng, rng = random.split(random.PRNGKey(42))
    params, vec_field_net = make_vec_field_net(init_rng)

    forward, batched_sampler, logp_fun = NeuralODE(vec_field_net, sample_size, n*dim)

    x0 = random.normal(rng, (sample_size, n*dim))

    print(forward(params, x0).shape)
    print(logp_fun(params, x0).shape)







