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


def NeuralODE(vec_field_net, sample_size, dim):
    
    def batched_sampler(rng, params):
        '''
        flow from x0 to x1
        '''
        x0 = random.normal(rng, (sample_size, dim))

        def _ode(x, t):
            return vmap(vec_field_net, (None, 0, None), 0)(params, x, t)

        xt = ode.odeint(_ode, 
                        x0, 
                        jnp.array([0.0, 1.0]),
                        rtol=1e-10, atol=1e-10
                        )
        return xt[-1]

    @partial(vmap, in_axes=(None, 0), out_axes=0)
    def logp(params, x):
        '''
        likelihood of given samples
        '''
        def base_logp(x):
            return norm.logpdf(x).sum(-1)

        def _ode(state, t):
            x = state[0]
            return -vec_field_net(params, x, t), divergence_fwd(vec_field_net)(params, x, t)

        logp = 0.0
        xt, logpt = ode.odeint(_ode, 
                            [x, logp], 
                            jnp.array([0.0, 1.0]),
                            rtol=1e-10, atol=1e-10
                            )
        return -logpt[-1] + base_logp(xt[-1])

    def divergence_fwd(f):
        def _div_f(params, x, t):
            jac = jax.jacfwd(lambda x: f(params, x, t))
            return jnp.trace(jac(x))
        return _div_f
    
    return batched_sampler, logp

if __name__ == '__main__':
    pass