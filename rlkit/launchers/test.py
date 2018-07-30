def check_exp_spec_format(specs):
    '''
        Check that all keys are strings that don't contain '.'
    '''
    for k, v in specs.items():
        if not isinstance(k, basestring): return False
        if '.' in k: return False
        if isinstance(v, dict):
            sub_ok = check_exp_spec_format(v)
            if not sub_ok: return False
    return True


def flatten_dict(dic):
    '''
        Assumes a potentially nested dictionary where all keys
        are strings that do not contain a '.'

        Returns a flat dict with keys having format:
        {'key.sub_key.sub_sub_key': ..., etc.} 
    '''
    new_dic = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            sub_dict = flatten_dict(v)
            for sub_k, v in sub_dict:
                new_dic['.'.join([k, sub_k])] = v
        else:
            new_dic[k] = v

    return new_dic


def add_variable_to_constant_specs(constants, flat_variables):
    new_dict = deepcopy(constants)
    for k, v in flat_variables:
        cur_sub_dict = new_dict
        split_k = k.split('.')
        for sub_key in split_k[:-1]: cur_sub_dict = cur_sub_dict[sub_key]
        cur_sub_dict[split_k[-1]] = v
    return new_dict


def build_nested_variant_generator(exp_spec):
    assert check_exp_spec_format(exp_spec)
    from rllab.misc.instrument import VariantGenerator

    variables = exp_spec['variables']
    constants = exp_spec['constants']

    variables = flatten_dict(variables)
    vg = VariantGenerator()
    for k, v in variables: vg.add(k, v)
    
    def vg_fn():
        for flat_variables in vg:
            yield add_variable_to_constant_specs(constants, flat_variables)

    return vg_fn


if __name__ == '__main__':
    variables = {
        'hi': {
            'one': [1,2,3,4],
            'two': [5678],
            'three': {
                'apple': ['yummy', 'sour', 'sweet']
            }
        },
        'bye': ['omg', 'lmfao', 'waddup']
    }

    constants = {
        'hi': {
            'three': {
                'constant_banana': 'potassium'
            },
            'other_constant_stuff': {
                'idk': 'something funny and cool'
            }
        },
        'yoyoyo': 'I like candy',
        'wow': 1e8
    }
