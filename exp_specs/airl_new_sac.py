meta_data:
  script_path: /u/kamyar/oorl_rlkit/run_scripts/on_the_fly_meta_sac.py
  exp_dirs: /ais/gobi6/kamyar/oorl_rlkit/output
  exp_name: final_pendulum_new_sac_hyper_param_search
  description: searching over the SAC hyperparameters
  num_workers: 10
  cpu_range: [0,10]
  num_cpu_per_worker: 1
# -----------------------------------------------------------------------------
variables:
  algo_params:
    concat_env_params_to_obs: [false]
    num_updates_per_env_step: [1]
    reward_scale: [5.0]
    soft_target_tau: [0.005]
  
  env_specs:
    normalized: [true, false]
  
  seed: [9783, 5914, 4865, 2135, 2349]

# -----------------------------------------------------------------------------
constants:
  algo_params:
    num_epochs: 51
    num_steps_per_epoch: 1000
    num_steps_per_eval: 1000
    batch_size: 256
    max_path_length: 100
    discount: 0.99

    policy_lr: 0.0003
    qf_lr: 0.0003
    vf_lr: 0.0003
    policy_mean_reg_weight: 0.001
    policy_std_reg_weight: 0.001

    render: false

    save_replay_buffer: true
  
  net_size: 256
  
  env_specs:
    base_env_name: 'pendulum_v0'
