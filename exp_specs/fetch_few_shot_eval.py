meta_data:
  script_path: /h/kamyar/oorl_rlkit/run_scripts/few_shot_recall_at_k_eval_script.py
  exp_dirs: /scratch/gobi2/kamyar/oorl_rlkit/output/
  exp_name: test_eval_recall_at_k
  description: searching over the SAC hyperparameters
  use_gpu: true
  num_workers: 4
  cpu_range: [0,159]
  num_cpu_per_worker: 1
# -----------------------------------------------------------------------------
variables:
  evaluating_np_airl: [false] # if false means evaluating np_bc
  sample_from_prior: [true]
  sub_exp: [
    final_better_np_bc_new_hype_search_KL_0p01_weight_share_enc_32_dim_pol_dim_64_gate_dim_32_z_dim_16_2019_02_19_20_37_53_0000--s-0,
    final_better_np_bc_new_hype_search_KL_0p01_weight_share_enc_32_dim_pol_dim_64_gate_dim_32_z_dim_16_2019_02_19_20_37_53_0001--s-0,
    final_better_np_bc_new_hype_search_KL_0p01_weight_share_enc_32_dim_pol_dim_64_gate_dim_32_z_dim_16_2019_02_19_20_37_54_0002--s-0,
    final_better_np_bc_new_hype_search_KL_0p01_weight_share_enc_32_dim_pol_dim_64_gate_dim_32_z_dim_16_2019_02_19_20_37_54_0003--s-0
  ]

# -----------------------------------------------------------------------------
constants:
  exp_path: /scratch/gobi2/kamyar/oorl_rlkit/output/final-better-np-bc-new-hype-search-KL-0p01-weight-share-enc-32-dim-pol-dim-64-gate-dim-32-z-dim-16
