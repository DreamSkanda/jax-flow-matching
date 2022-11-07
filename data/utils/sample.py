import jax.numpy as jnp
import numpy as np
from jax import random, vmap
from energy import energy_func
from mcmc import mcmc_func

from functools import partial
import matplotlib.pyplot as plt

if __name__ == '__main__':

    batch_size = 10240
    n = 6
    dim = 2
    beta = 10

    mc_steps = 100 
    mc_width = 0.05

    init_rng, rng = random.split(random.PRNGKey(42))
    x = random.normal(init_rng, (batch_size, n, dim))

    @partial(vmap, in_axes=(None, 0, None, None))
    def logp(beta, x, n, dim):
        return -beta * energy_func(x, n, dim)

    for _ in range(20):
        mcmc_rng, rng = random.split(rng)
        x, acc = mcmc_func(lambda x: logp(beta, x, n, dim), x, mcmc_rng, mc_steps, mc_width)
        e = vmap(energy_func, (0, None, None), 0)(x, n, dim)
        print (acc, jnp.mean(e), jnp.std(e)/jnp.sqrt(batch_size))

    x = jnp.reshape(x, (batch_size*n, dim)) 
    #density plot
    H, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], 
                                        bins=100, 
                                        range=((-4, 4), (-4, 4)),
                            density=True)

    plt.imshow(H, interpolation="nearest", 
                extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                cmap="inferno")