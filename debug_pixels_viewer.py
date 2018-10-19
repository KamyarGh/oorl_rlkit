import joblib
from scipy.misc import imsave

d = joblib.load(
    '/ais/gobi6/kamyar/oorl_rlkit/output/'
    + 'dmcs-reacher-new-sac-rew-scale-10/dmcs_reacher_new_sac_rew_scale_10_2018_10_19_08_49_55_0004--s-0/extra_data.pkl')

for i in range(1000):
    imsave('plots/test_pixels/%d.png'%i, d['replay_buffer']._observations['pixels'][i])
