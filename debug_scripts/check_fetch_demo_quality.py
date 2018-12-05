import numpy as np


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

her_demos_path = '/u/kamyar/baselines/baselines/her/data_fetch_random_100.npz'
d = np.load(her_demos_path)

rews = []
path_lens = []
for path in d['obs']:
    path_rew = 0
    # for step in path:
    for i in range(50):
        step = path[i]
        ag = step['achieved_goal']
        dg = step['desired_goal']
        dist = goal_distance(ag, dg)
        if dist > 0.05:
            path_rew += -1.0
        else:
            path_rew += -1.0*dist
    rews.append(path_rew)
    # path_lens.append(len(path))
    path_lens.append(50)

zipped = list(zip(rews, path_lens))
print(zipped)
solved = [t[0] > -1.0*t[1] for t in zipped]
print(solved)
print('%.4f +/- %.4f' % (np.mean(rews), np.std(rews)))
print(sum(solved))
print(sum(solved) / float(len(solved)))

# compute action stats
all_acts = np.array([
    a for path in d['acs'] for a in path
])
print(all_acts.shape)
print(np.mean(all_acts, axis=0))
# [-0.00097687 -0.00931541  0.00991785  0.01412615]
print(np.std(all_acts, axis=0))
# [0.27218603 0.25925623 0.32518755 0.02619406]
abs_acts = np.abs(all_acts)
print(np.mean(abs_acts, axis=0))
# [0.14009708 0.13060458 0.13870633 0.02064866]
print(np.std(abs_acts, axis=0))
# [0.23336452 0.22414953 0.29428874 0.0214315 ]
