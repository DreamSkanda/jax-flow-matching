import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.experimental import ode
from diffrax import diffeqsolve, ODETerm, Dopri5
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


def NeuralODE(vec_field_net, sample_size, dim):
    
    def batched_sampler(rng, params):
        '''
        flow from x0 to x1
        '''
        def f(t, y, args):
            return vmap(vec_field_net, (None, 0, None), 0)(params, y, t)

        term = ODETerm(f)
        solver = Dopri5()
        y0 = random.normal(rng, (sample_size, dim))
        solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0, max_steps=100)

        return solution.ys

    @partial(vmap, in_axes=(None, 0), out_axes=0)
    def logp(params, x):
        '''
        likelihood of given samples
        '''
        def base_logp(x):
            return norm.logpdf(x).sum(-1)

        def f(t, y, args):
            return [-vec_field_net(params, y[0], t), divergence_fwd(vec_field_net)(params, y[0], t)]

        logp = 0.0

        term = ODETerm(f)
        solver = Dopri5()
        solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=[x, logp], max_steps=100)
        
        return - solution.ys[1] + base_logp(solution.ys[0])

    def divergence_fwd(f):
        def _div_f(params, x, t):
            jac = jax.jacfwd(lambda x: f(params, x, t))
            return jnp.trace(jac(x))
        return _div_f
    
    return batched_sampler, logp

if __name__ == '__main__':
    from jax.example_libraries.stax import serial, Dense, Relu
    from jax.nn.initializers import zeros

    dim = 2
    sample_size = 10

    def make_vec_field_net(rng):
        net_init, net_apply = serial(Dense(8), Dense(dim, W_init=zeros, b_init=zeros))
        in_shape = (-1, dim+1)
        _, net_params = net_init(rng, in_shape)

        def net_with_t(params, x, t):
            return net_apply(params, jnp.concatenate((x,t.reshape(1))))
        
        return net_params, net_with_t

    init_rng, rng = random.split(random.PRNGKey(42))
    params, vec_field_net = make_vec_field_net(init_rng)

    batched_sampler, batched_logp = NeuralODE(vec_field_net, sample_size, dim)

    print(batched_sampler(rng, params).shape)

