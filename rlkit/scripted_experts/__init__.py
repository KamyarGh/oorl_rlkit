from rlkit.scripted_experts.few_shot_fetch_env_expert import ScriptedFewShotFetchPolicy

_pantry = {
    'few_shot_fetch_scripted_policy': lambda: ScriptedFewShotFetchPolicy()
}

def get_scripted_policy(scripted_policy_name):
    return _pantry[scripted_policy_name]()
