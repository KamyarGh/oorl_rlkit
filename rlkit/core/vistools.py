import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_returns_on_same_plot(arr_list, names, title, save_path):
    fig, ax = plt.subplots(1)
    cmap = get_cmap(len(arr_list))

    for i, v in enumerate(zip(arr_list, names)):
        ret, name = v
        ax.plot(np.arange(ret.shape[0]), ret, color=cmap(i), label=name)

    ax.set_title(title)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=3)
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def save_plot(x, y, title, save_path, color='cyan'):
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(x, y, color=color))
    ax.set_title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
