from multiprocessing import Pool
import yaml
import argparse
from rlkit.launchers.launcher_util import setup_logger, build_nested_variant_generator

# from exp_pool_fns.neural_process_v1 import exp_fn


def get_pool_function(exp_fn_name):
    if exp_fn_name == 'neural_processes_v1':
        from exp_pool_fns.neural_process_v1 import exp_fn
    
    return exp_fn


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    vg_fn = build_nested_variant_generator(exp_specs)
    all_exp_args = []
    for i, variant in enumerate(vg_fn()):
        all_exp_args.append([variant, i])

    num_total = len(all_exp_args)
    num_workers = min(exp_specs['meta_data']['num_workers'], num_total)
    p = Pool(num_workers)
    pool_function = get_pool_function(exp_specs['meta_data']['exp_fn_name'])
    print(
        '\n\n\n\n{}/{} experiments ran successfully!'.format(
            sum(p.map(pool_function, all_exp_args)),
            num_total
        )
    )
