import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_histogram(flat_array, num_bins, title, save_path):
    fig, ax = plt.subplots(1)
    ax.set_title(title)
    plt.hist(flat_array, bins=num_bins)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_pytorch_tensor_as_img(tensor, save_path):
    if tensor.size(0) == 1: tensor = tensor.repeat(3, 1, 1)
    fig, ax = plt.subplots(1)
    ax.imshow(np.transpose(tensor.numpy(), (1,2,0)))
    plt.savefig(save_path)
    plt.close()


def generate_gif(list_of_img_list, names, save_path):
    fig, axarr = plt.subplots(len(list_of_img_list))
    def update(t):
        for j in range(len(list_of_img_list)):
            axarr[j].imshow(list_of_img_list[j][t])
            axarr[j].set_title(names[j])
        return axarr
    anim = FuncAnimation(fig, update, frames=np.arange(len(list_of_img_list[0])), interval=2000)
    anim.save(save_path, dpi=80, writer='imagemagick')
    plt.close()


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    # for some weird reason 0 and 2 look almost identical
    n += 1
    cmap = plt.cm.get_cmap(name, n)
    # def new_cmap(n):
    #     if n >= 3: n = n+1
    #     if n == 1:
    #         return (0,0,0,1)
    #     else:
    #         return cmap(n)
    # return new_cmap
    return cmap


def plot_returns_on_same_plot(arr_list, names, title, save_path, x_axis_lims=None, y_axis_lims=None):
    # print(arr_list, names, title, save_path, y_axis_lims)
    fig, ax = plt.subplots(1)
    cmap = get_cmap(len(arr_list))
    for i in range(len(arr_list)): cmap(i)

    for i, v in enumerate(zip(arr_list, names)):
        ret, name = v
        if ret.size <= 1: continue
        ax.plot(np.arange(ret.shape[0]), ret, color=cmap(i), label=name)

    ax.set_title(title)
    if x_axis_lims is not None:
        ax.set_xlim(x_axis_lims)
    if y_axis_lims is not None:
        ax.set_ylim(y_axis_lims)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=3)
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_multiple_plots(plot_list, names, title, save_path):
    fig, ax = plt.subplots(1)
    cmap = get_cmap(len(plot_list))

    for i, v in enumerate(zip(plot_list, names)):
        plot, name = v
        ax.plot(plot[0], plot[1], color=cmap(i), label=name)

    ax.set_title(title)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=3)
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def save_plot(x, y, title, save_path, color='cyan', x_axis_lims=None, y_axis_lims=None):
    fig, ax = plt.subplots(1)
    ax.plot(x, y, color=color)
    ax.set_title(title)
    if x_axis_lims is not None:
        ax.set_xlim(x_axis_lims)
    if y_axis_lims is not None:
        ax.set_ylim(y_axis_lims)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_forward_reverse_KL_rews():
    # reverse KL
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(-10,10,0.05), np.arange(-10,10,0.05), color='cyan')
    ax.set_xlim([-10,10])
    ax.set_ylim([-12,12])
    ax.set_xlabel(r'log$\frac{\rho^{exp}(s,a)}{\rho^\pi(s,a)}$', fontsize='xx-large')
    ax.set_ylabel('$r(s,a)$', fontsize='xx-large')
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.savefig('plots/junk_vis/rev_KL_rew.png', bbox_inches='tight', dpi=150)
    plt.close()

    # forward KL
    fig, ax = plt.subplots(1)
    x = np.arange(-10,10,0.05)
    y = np.exp(x) * (-x)
    ax.plot(x, y, color='cyan')
    ax.set_xlim([-10,10])
    ax.set_ylim([-2,0.5])
    ax.set_xlabel(r'log$\frac{\rho^{exp}(s,a)}{\rho^\pi(s,a)}$', fontsize='xx-large')
    ax.set_ylabel('$r(s,a)$', fontsize='xx-large')
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.savefig('plots/junk_vis/forw_KL_rew.png', bbox_inches='tight', dpi=150)
    plt.close()
