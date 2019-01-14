from rlkit.scripted_experts.few_shot_fetch_env_expert import ScriptedFewShotFetchPolicy
from rlkit.scripted_experts.cont_few_shot_fetch_env_expert import ScriptedContFewShotFetchPolicy
from rlkit.scripted_experts.linear_few_shot_fetch_env_expert import ScriptedLinearFewShotFetchPolicy

from rlkit.scripted_experts.linear_few_shot_reach_env_expert import ScriptedLinearFewShotReachPolicy

_pantry = {
    'few_shot_fetch_scripted_policy': lambda: ScriptedFewShotFetchPolicy(),
    'cont_few_shot_fetch_scripted_policy': lambda: ScriptedContFewShotFetchPolicy(),
    'linear_few_shot_fetch_scripted_policy': lambda: ScriptedLinearFewShotFetchPolicy(),
    'linear_few_shot_reach_scripted_policy': lambda: ScriptedLinearFewShotReachPolicy(),
}

def get_scripted_policy(scripted_policy_name):
    return _pantry[scripted_policy_name]()
