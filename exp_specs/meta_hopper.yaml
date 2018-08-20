meta_data:
  script_path: /u/kamyar/oorl_rlkit/run_scripts/on_the_fly_meta_sac.py
  exp_dirs: /u/kamyar/oorl_rlkit/output
  exp_name: hopper_on_the_fly
  description: searching over the SAC hyperparameters
  num_workers: 8
  cpu_range: [0,7]
  num_cpu_per_worker: 1

# -----------------------------------------------------------------------------
variables:
  use_new_sac: [false]
  algo_params:
    epoch_to_start_training: [50]
    concat_env_params_to_obs: [true, false]
    normalize_env_params: [true]
    num_updates_per_env_step: [4]
    reward_scale: [1]
    soft_target_tau: [0.005]
  
  env_specs:
    normalized: [true, false]
  
  # seed: [9783, 5914, 4865, 2135, 2349]
  seed: [9783, 5914]

# -----------------------------------------------------------------------------
constants:
  algo_params:
    num_epochs: 1001
    num_steps_per_epoch: 1000
    num_steps_per_eval: 1000
    batch_size: 128
    max_path_length: 999
    discount: 0.99

    policy_lr: 0.0003
    qf_lr: 0.0003
    vf_lr: 0.0003

    render: false

    save_replay_buffer: true

  net_size: 128

  env_specs:
    base_env_name: 'meta_gears_hopper'
    gear_0: [50, 200.0]
    gear_1: [50, 200.0]
    gear_2: [50, 200.0]
