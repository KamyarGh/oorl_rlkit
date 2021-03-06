import joblib
import numpy as np
from rlkit.core.vistools import plot_2dhistogram, plot_scatter, plot_seaborn_heatmap

def gen_plus_sign(distance, num_points_per_direction, save_path):
    SCALE = 1.0
    # CENTER_BUFFER = 3*SCALE
    # CENTER_BUFFER = 6*SCALE
    CENTER_BUFFER = 0.0
    print(SCALE, CENTER_BUFFER)
    X = []
    Y = []

    # East
    for _ in range(num_points_per_direction):
        x = np.random.uniform(CENTER_BUFFER, float(distance))
        y = np.random.normal(loc=0.0, scale=SCALE)
        X.append(x)
        Y.append(y)

    # West
    for _ in range(num_points_per_direction):
        x = np.random.uniform(-float(distance), -CENTER_BUFFER)
        y = np.random.normal(loc=0.0, scale=SCALE)
        X.append(x)
        Y.append(y)

    # North
    for _ in range(num_points_per_direction):
        y = np.random.uniform(CENTER_BUFFER, float(distance))
        x = np.random.normal(loc=0.0, scale=SCALE)
        X.append(x)
        Y.append(y)

    # South
    for _ in range(num_points_per_direction):
        y = np.random.uniform(-float(distance), -CENTER_BUFFER)
        x = np.random.normal(loc=0.0, scale=SCALE)
        X.append(x)
        Y.append(y)

    plot_2dhistogram(
        X, Y, 30, 'test', 'plots/junk_vis/test_plus.png', [[-distance,distance], [-distance,distance]]
    )
    plot_scatter(
        X, Y, 30, 'test', 'plots/junk_vis/test_plus_scatter.png', [[-distance,distance], [-distance,distance]]
    )
    plot_seaborn_heatmap(
        X, Y, 30, 'test', 'plots/junk_vis/test_plus_sns.png', [[-distance,distance], [-distance,distance]]
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

if __name__ == '__main__':
    gen_plus_sign(8.0, 3200, save_path='/scratch/hdd001/home/kamyar/expert_demos/xy_pos_data/test.pkl')
    # gen_plus_sign(8.0, 3200, save_path='/scratch/hdd001/home/kamyar/expert_demos/xy_pos_data/plus_dist_8_scale_0p5_no_buffer_3200_per_dir.pkl')
    # gen_plus_sign(4.0, 400)
    # gen_plus_sign(2.0, 800)
