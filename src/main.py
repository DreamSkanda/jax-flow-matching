import jax
import jax.numpy as jnp

from data import make_sampler
from net import make_vec_field_net, make_backflow, make_transformer
from flow import NeuralODE
from loss import make_loss
from energy import energy_fun, make_free_energy
from train import train, train_and_evaluate
from plot import plot

import argparse
import time
import os

jax.config.update("jax_enable_x64", True)
rng = jax.random.PRNGKey(42)

####################################################################################

parser = argparse.ArgumentParser(description="")

parser.add_argument("--plot", action="store_true", help="Plot after training")

group = parser.add_argument_group("learning parameters")
group.add_argument("--coupled", action="store_true", help="Use coupled training method")
group.add_argument("--epochs", type=int, default=1000, help="Epochs for training")
group.add_argument("--iterations", type=int, default=50, help="Training iterations in an epoch")
group.add_argument("--batchsize", type=int, default=4096, help="")
group.add_argument("--samplesize", type=int, default=8192, help="")
group.add_argument("--lr", type=float, default=1e-3, help="learning rate")
group.add_argument("--data", default="../data/", help="The folder to save data")

group = parser.add_argument_group("datasets")
group.add_argument("--dataset", default="../datasets/",help="The folder to load training datasets")

group = parser.add_argument_group("network parameters")
group.add_argument("--nhiddens", type=int, default=512, help="The channels in a middle layer")
group.add_argument("--nlayers", type=int, default=4, help="The number of layers")
group.add_argument("--nheads", type=int, default=8, help="")
group.add_argument("--keysize", type=int, default=16, help="")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--backflow", action="store_true", help="Use backflow")
group.add_argument("--transformer", action="store_true", help="Use transformer")
group.add_argument("--mlp", action="store_true", help="Use mlp")
group.add_argument("--emlp", action="store_true", help="Use equivariant-mlp")

group = parser.add_argument_group("physics parameters")
group.add_argument("--n", type=int, default=6, help="The number of particles")
group.add_argument("--dim", type=int, default=2, help="The dimensions of the system")
group.add_argument("--beta", type=float, default=10.0, help="The inverse temperature")

args = parser.parse_args()

datasize = round(1.5*args.batchsize)*args.iterations if args.coupled else args.batchsize*args.iterations

####################################################################################

print("\n========== Prepare training dataset ==========")

os.makedirs(args.dataset, exist_ok=True)

ds_filename = os.path.join(args.dataset, "datasize_%d_n_%d_dim_%d_beta_%g.npz" % (datasize, args.n, args.dim, args.beta))

if os.path.isfile(ds_filename):
    data = jnp.load(ds_filename)
    X0, X1 = data["X0"], data["X1"]

    print("Load dataset: %s" % ds_filename)
else:   
    sampler = make_sampler(datasize)
    data_rng, rng = jax.random.split(rng)

    start = time.time()
    X0, X1 = sampler(data_rng, args.beta, args.n, args.dim)
    end = time.time()
    running_time = end - start
    jnp.savez(ds_filename, X0=X0, X1=X1)      

    print("Generate dataset: %s" % ds_filename)
    print("sampling time: %.5f sec" % running_time)

####################################################################################

init_rng, rng = jax.random.split(rng)
if args.backflow:
    print("\n========== Construct backflow network ==========")
    params, vec_field_net = make_backflow(init_rng, args.n, args.dim, [args.nhiddens]*args.nlayers)
    modelname = "backflow_nl_%d_nh_%d" % (args.nlayers, args.nhiddens)
elif args.transformer:
    print("\n========== Construct transformer network ==========")
    params, vec_field_net = make_transformer(init_rng, args.n, args.dim, args.nheads, args.nlayers, args.keysize)
    modelname = "transformer_nl_%d_nh_%d_nk_%d" % (args.nlayers, args.nheads, args.keysize)
elif args.mlp:
    print("\n========== Construct mlp network ==========")
    params, vec_field_net = make_vec_field_net(init_rng, args.n, args.dim, ch=args.nhiddens, num_layers=args.nlayers, symmetry=False)
    modelname = "mlp_nl_%d_nh_%d" % (args.nlayers, args.nhiddens)
elif args.emlp:
    print("\n========== Construct emlp network ==========")
    params, vec_field_net = make_vec_field_net(init_rng, args.n, args.dim, ch=args.nhiddens, num_layers=args.nlayers, symmetry=True)
    modelname = "emlp"
else:
    raise ValueError("what model ?")

"""initializing the sampler and logp calculator"""
_, _, batched_sample_fun, _ = NeuralODE(vec_field_net, args.n*args.dim)
free_energy = make_free_energy(energy_fun, batched_sample_fun, args.n, args.dim, args.beta)

"""initializing the loss function"""
loss = make_loss(vec_field_net)
value_and_grad = jax.value_and_grad(loss)

####################################################################################

print("\n========== Prepare logs ==========")

methodname = "coupled" if args.coupled else "discrete"
path = args.data + "n_%d_dim_%d_beta_%g_lr_%g" % (args.n, args.dim, args.beta, args.lr) \
                    + "_" + methodname + "_" + modelname
os.makedirs(path, exist_ok=True)
print("Create directory: %s" % path)

####################################################################################

print("\n========== Train with %s samples ==========" % methodname)

hyperparams = (args.epochs, args.iterations, args.batchsize)
start = time.time()
if args.coupled:
    training_data = (X0[0:args.iterations*args.batchsize], X1[0:args.iterations*args.batchsize])
    validation_data = (X0[args.iterations*args.batchsize:datasize], X1[args.iterations*args.batchsize:datasize])
    params = train_and_evaluate(rng, loss, value_and_grad, hyperparams, params, training_data, validation_data, args.lr, path)
else:
    params = train(rng, value_and_grad, hyperparams, params, X1, args.lr, path)
end = time.time()
running_time = end - start
print("training time: %.5f sec" %running_time)

####################################################################################

print("\n========== Draw samples and calculate free energy ==========")

start = time.time()
fe_rng, rng = jax.random.split(rng)
fe, fe_err, samples, f, f_err = free_energy(fe_rng, params, args.samplesize)
end = time.time()
running_time = end - start
print("free energy: %f ± %f" %(fe, fe_err))
print("variational free energy: %f ± %f" %(f, f_err))
print("importance sampling time: %.5f sec" %running_time)

smp_filename = os.path.join(path, "samples.npy")
jnp.save(smp_filename, samples)

if args.plot:
    plot(path, args.coupled)