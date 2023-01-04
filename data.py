from jax import random
from mcmc import mcmc_fun
from energy import energy_fun

def make_sampler(data_size: int):

    def sampler(rng, beta, n, dim, mc_epoch=20, mc_steps=100, mc_width=0.05):
        
        sample_rng, rng = random.split(rng)
        X = random.normal(sample_rng, (data_size, n*dim))
        X0 = X

        for _ in range(mc_epoch):
            mcmc_rng, rng = random.split(rng)
            X, acc = mcmc_fun(mcmc_rng, lambda X: -beta*energy_fun(X, n, dim), X, mc_steps, mc_width)
            e = energy_fun(X, n, dim)
        
        X1 = X

        return X0, X1

    return sampler