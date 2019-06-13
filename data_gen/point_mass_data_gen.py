import numpy as np
import joblib
import os.path as osp
from rlkit.core.vistools import plot_2dhistogram, plot_scatter, plot_seaborn_heatmap

def line_data(length, width_scale, num_points):
    X = np.random.normal(loc=0.0, scale=width_scale, size=num_points)
    Y = 2*length*np.random.uniform(size=num_points) - length
    return X, Y


def two_gaussians(distance, scale, num_points, lower_only=False):
    X, Y = [], []
    for _ in range(num_points):
        if lower_only:
            c = np.array([0.0, -distance])
        else:
            if np.random.uniform() > 0.5:
                c = np.array([0.0, distance])
            else:
                c = np.array([0.0, -distance])
        noise = np.random.normal(scale=scale, size=2)
        X.append(c[0] + noise[0])
        Y.append(c[1] + noise[1])
    return np.array(X), np.array(Y)


def four_gaussians(distance, scale, num_points):
    X, Y = [], []
    for _ in range(num_points):
        if np.random.uniform() > 0.5:
            if np.random.uniform() > 0.5:
                c = np.array([0.0, distance])
            else:
                c = np.array([0.0, -distance])
        else:
            if np.random.uniform() > 0.5:
                c = np.array([distance, 0.0])
            else:
                c = np.array([-distance, 0.0])
        noise = np.random.normal(scale=scale, size=2)
        X.append(c[0] + noise[0])
        Y.append(c[1] + noise[1])
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    # length = 3.0
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/line_len_3_scale_0p2_4000_points.pkl'
    # X, Y = line_data(length, 0.2, 4000)

    # dist = 3.0
    # scale = 0.2
    # save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/lower_only_two_gaussians_dist_3_scale_0p2_4000_points.pkl'
    # X, Y = two_gaussians(dist, scale, 4000, lower_only=True)

    dist = 3.0
    scale = 0.2
    save_path = '/scratch/hdd001/home/kamyar/expert_demos/data_gen/four_gaussians_dist_3_scale_0p2_4000_points.pkl'
    X, Y = four_gaussians(dist, scale, 4000)

    bound = 5
    plot_2dhistogram(
        X, Y, 30, 'test', 'plots/data_gen/hist.png', [[-bound,bound], [-bound,bound]]
    )
    plot_scatter(
        X, Y, 30, 'test', 'plots/data_gen/scatter.png', [[-bound,bound], [-bound,bound]]
    )
    plot_seaborn_heatmap(
        X, Y, 30, 'test', 'plots/data_gen/heatmap.png', [[-bound,bound], [-bound,bound]]
    )

    XY_DATA = np.array([X,Y]).T
    print(XY_DATA.shape)
    print(np.mean(XY_DATA, axis=0).shape)
    print(np.std(XY_DATA, axis=0).shape)
    joblib.dump(
        {
            'xy_data': XY_DATA,
            'xy_mean': np.mean(XY_DATA, axis=0),
            'xy_std': np.std(XY_DATA, axis=0)
        },
        save_path,
        compress=3
    )
