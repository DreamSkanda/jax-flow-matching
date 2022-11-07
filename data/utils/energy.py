import numpy as np
import jax.numpy as jnp
from jax import vmap
from functools import partial

@partial(vmap, in_axes=(0, None, None))
def energy_func(x, n, dim):
    i, j = np.triu_indices(n, k=1)
    r_ee = jnp.linalg.norm((jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))[i,j], axis=-1)
    v_ee = jnp.sum(1/r_ee)
    return jnp.sum(x**2) + v_ee