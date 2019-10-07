# Guide to reproducing the results from the fmax paper
Notes:
- First appropriately modify rlkit/launchers/config.py
- run_experiment.py calls srun which is a SLURM command. You can use the `--nosrun` flag to not use SLURM and use your local machine instead.
- The expert demonstrations and state marginal data used for imitation learning experiments can be found at [THIS LINK](https://drive.google.com/drive/folders/1M8XvJrMU24Hq_OMGR_SylsBrsSjY4WFd?usp=sharing).
- The yaml files describe the experiments to run and have three sections:
..* meta_data: general experiment and resource settings
..* variables: used to describe the hyperparameters to search over
..* constants: hyperparameters that will not be searched over


## Reproducing Imitation Learning Results
### Training Expert Policies
If you would like train your own expert policies with Soft-Actor-Critic you can for example run:
```bash
python run_experiment.py -e exp_specs/sac.yaml
```
To train a policy for a different environment, add your environment to the file in rlkit/envs/envs_dict and replace the name of your environment in the env_specs->env_name field in sac.yaml.

### Generating Demonstrations Using an Expert Policy
Modify exp_specs/gen_exp_demos.yaml appropriately and run:
```bash
python run_experiment.py -e exp_specs/gen_exp_demos.yaml
```

### Training F/AIRL
Put the path to your expert demos in expert_demos_listing.yaml. Modify exp_specs/adv_irl.yaml appropriately and run:
```bash
python run_experiment.py -e exp_specs/adv_irl.yaml
```
For all four imitation learning domains we used the same hyperparameters except grad_pen_weight and reward_scale which were chosen with a small hyperparameter search.

### Training BC
```bash
python run_experiment.py -e exp_specs/bc.yaml
```

### Training DAgger
```bash
python run_experiment.py -e exp_specs/dagger.yaml
```

## Reproducing State-Marginal-Matching Results
### Generating the target state marginal distributions
This is a little messy and I haven't gotten to cleaning it up yet. All the scripts that you see in the appendix of the paper can be found in these three files: data_gen/point_mass_data_gen.py, data_gen/fetch_state_marginal_matching_data_gen.py, data_gen/pusher_data_gen.py.

### Training SMM Models
```bash
python run_experiment.py -e exp_specs/adv_smm.yaml
```
