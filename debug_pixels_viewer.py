import joblib
from scipy.misc import imsave

d = joblib.load(
    '/ais/gobi6/kamyar/oorl_rlkit/output/'
    + 'dmcs-reacher-hype-search/dmcs_reacher_hype_search_2018_10_22_16_46_30_0003--s-0/extra_data.pkl')

# print(d['replay_buffer']._observations)

for i in range(1000):
    imsave('plots/test_pixels/%d.png'%i, d['replay_buffer']._observations['pixels'][i])
