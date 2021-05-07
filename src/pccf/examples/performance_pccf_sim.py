import random
import matplotlib.pyplot as plt
from pccf.pccf_performance import *


if __name__ == "__main__":
    mu_true = 10
    changes = [mu_true * i + random.randint(-2, 2) for i in range(1, 100)]
    save_plot = './img/pccf_performance_sim.PNG'
    mu_vec, f1_vec = collect_pccf_performance(changes, mu_range=(1, 20), radius_range=(3, 7))
    fig = plt.figure(111, figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.plot(mu_vec, f1_vec)
    ax.set_title("Pccf performance for ideal changes")
    ax.set_xlabel('mu estimation')
    ax.set_ylabel('F1')
    print(f"Save into {save_plot}")
    plt.savefig(save_plot)
