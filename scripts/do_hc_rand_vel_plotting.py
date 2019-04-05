import joblib
from collections import defaultdict
from os import path as osp

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

expert_path = '/scratch/gobi2/kamyar/oorl_rlkit/output/test/test_2019_03_29_00_19_09_0000--s-0'
rb = joblib.load(osp.join(expert_path, 'extra_data.pkl'))['meta_train']['context']
X, means, stds = [], [], []
for t in sorted(list(rb.task_replay_buffers.keys())):
    s = rb.task_replay_buffers[t]
    splits = np.split(s._rewards[:s._size], 4)
    # ret_means = np.mean(list(map(lambda p: np.sum(p[:50]), splits)))
    # ret_stds = np.std(list(map(lambda p: np.sum(p[:50]), splits)))
    ret_means = np.mean(list(map(lambda p: np.mean(p[:50]), splits)))
    ret_stds = np.mean(list(map(lambda p: np.std(p[:50]), splits)))
    X.append(t)
    means.append(ret_means)
    stds.append(ret_stds)
X = np.array(X)
means = np.array(means)
stds = np.array(stds)

fig, ax = plt.subplots(1)
ax.plot(X, means)
ax.plot(X, means + stds)
ax.plot(X, means - stds)
# avg = np.mean(means)
# ax.plot(X, [avg for _ in range(len(means))])
ax.plot(X,X)
plt.savefig('plots/junk_vis/vels_for_hc_rand_vel_expert.png', bbox_inches='tight', dpi=300)
plt.close()
