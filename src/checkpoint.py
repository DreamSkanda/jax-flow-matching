import pickle
import os
import re

def find_ckpt_filename(path_or_file):
    if os.path.isfile(path_or_file):
        epoch = int(re.search("epoch_([0-9]*).pkl", path_or_file).group(1))
        return path_or_file, epoch
    files = [f for f in os.listdir(path_or_file) if ('pkl' in f)]
    for f in sorted(files, reverse=True):
        fname = os.path.join(path_or_file, f)
        try:
            with open(fname, "rb") as f:
                pickle.load(f)
            epoch = int(re.search("epoch_([0-9]*).pkl", fname).group(1))
            return fname, epoch
        except (OSError, EOFError):
            print("Error loading checkpoint. Trying next checkpoint...", fname)
    return None, 0

def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def ckpt_hyperparams(filename):
    if os.path.isfile(filename):
        n = int(re.search(r"/n_([0-9]*)_", filename).group(1))
        dim = int(re.search(r"_dim_([0-9]*)_", filename).group(1))
        beta = float(re.search(r"_beta_(.*?)_", filename).group(1))

        net = re.search(r"_([^_]*?)_nl_", filename).group(1)
        nlayers = int(re.search(r"_nl_([0-9]*)_", filename).group(1))
        nh = int(re.search(r"_nh_([0-9]*)", filename).group(1))
        _ = re.search(r"_nk_([0-9]*)", filename)
        nk = int(_.group(1)) if _ else None

        return (n, dim, beta), (net, nlayers, nh, nk)