import argparse
import joblib

assert False, 'I have not successfully used or tested this yet'

def experiment(checkpoint):
    algorithm = joblib.load(checkpoint)['algorithm']
    algorithm.render = True
    algorithm.do_not_train = True
    algorithm.do_not_eval = True

    if ptu.gpu_enabled():
        algorithm.cuda()
    else:
        algorithm.cpu()
    algorithm.train()

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='experiment specification file')
    args = parser.parse_args()

    experiment(args.checkpoint)
