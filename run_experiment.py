import multiprocessing
from multiprocessing import Pool, Process
import yaml
import argparse
import psutil
import os
from queue import Queue
from time import sleep
from subprocess import Popen
import datetime
import dateutil
from rlkit.launchers.launcher_util import setup_logger, build_nested_variant_generator

# from exp_pool_fns.neural_process_v1 import exp_fn


def get_pool_function(exp_fn_name):
    if exp_fn_name == 'neural_processes_v1':
        from exp_pool_fns.neural_process_v1 import exp_fn
    elif exp_fn_name == 'sac':
        from exp_pool_fns.sac import exp_fn
    
    return exp_fn


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    # generating the variants
    vg_fn = build_nested_variant_generator(exp_specs)
    # all_exp_args = []
    # for i, variant in enumerate(vg_fn()):
    #     all_exp_args.append([variant, i])
    
    # write all of them to a file
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    variants_dir = os.path.join(
        exp_specs['meta_data']['exp_dirs'], exp_specs['meta_data']['exp_name'], 'variants-'+timestamp
    )
    os.makedirs(variants_dir)
    with open(os.path.join(variants_dir, 'exp_spec_definition.yaml'), 'w') as f:
        yaml.dump(exp_specs, f, default_flow_style=False)
    num_variants = 0
    for variant in vg_fn():
        i = num_variants
        variant['exp_id'] = i
        with open(os.path.join(variants_dir, '%d.yaml'%i), 'w') as f:
            yaml.dump(variant, f, default_flow_style=False)
            f.flush()
        num_variants += 1
    
    # setting up pool and cpu affinity
    num_workers = min(exp_specs['meta_data']['num_workers'], num_variants)

    cpu_range = exp_specs['meta_data']['cpu_range']
    num_available_cpus = cpu_range[1] - cpu_range[0] + 1
    num_cpu_per_worker = exp_specs['meta_data']['num_cpu_per_worker']
    assert  num_cpu_per_worker * num_workers <= num_available_cpus

    affinities = []
    for i in range(int(num_available_cpus / num_cpu_per_worker)):
        affinities.append(
            [
                cpu_range[0] + num_cpu_per_worker * i + j
                for j in range(num_cpu_per_worker)
            ]
        )
    affinities = [hex(sum(2**i for i in aff)) for aff in affinities]
    affinity_Q = Queue()
    for aff in affinities: affinity_Q.put(aff)

    # run the processes
    running_processes = {}
    args_idx = 0
    command = 'taskset {aff} python {script} -e {specs}'
    while (args_idx < num_variants) or (len(running_processes) > 0):
        if (len(running_processes) < num_workers) and (args_idx < num_variants):
            aff = affinity_Q.get()
            format_dict = {
                'aff': aff,
                'script': exp_specs['meta_data']['script_path'],
                'specs': os.path.join(variants_dir, '%i.yaml'%args_idx)
            }
            command_to_run = command.format(**format_dict)
            command_to_run = command_to_run.split()
            p = Popen(command_to_run)
            args_idx += 1
            running_processes[p] = aff
        
        new_running_processes = {}
        for p, aff in running_processes.items():
            ret_code = p.poll()
            if ret_code is None:
                new_running_processes[p] = aff
            else:
                affinity_Q.put(aff)
        running_processes = new_running_processes
        
        sleep(0.5)
