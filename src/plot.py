import numpy as np
import matplotlib.pyplot as plt
import os

def plot(path, coupled):

    print("\n========== Start plotting ==========")

    smp_filename = os.path.join(path, "samples.npy")
    samples = np.load(smp_filename).reshape(-1, 2)
    
    log_filename = os.path.join(path, "loss.txt")
    loss = np.loadtxt(log_filename)

    plot_range = [(-2, 2), (-2, 2)]
    n_bins = 100

    fig = plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.hist2d(samples[:, 0], samples[:, 1], bins=n_bins, range=plot_range)
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(loss.shape[0]), loss[:, 1], label="train loss")
    if coupled:
        plt.plot(np.arange(loss.shape[0]), loss[:, 2], label="validation loss")
    plt.legend()

    fig_filename = os.path.join(path, "figure.png")
    plt.savefig(fig_filename)
    print("Save figure: %s" % fig_filename)

if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--folder", help="The folder to load samples")
    args = parser.parse_args()
    
    match = re.search(r"coupled", args.folder)

    plot(args.folder, bool(match))