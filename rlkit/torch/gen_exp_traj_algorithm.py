from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.data_management.expert_replay_buffer import ExpertReplayBuffer

class ExpertTrajGeneratorAlgorithm(TorchRLAlgorithm):
    def __init__(self, *args, **kwargs):
        super(ExpertTrajGeneratorAlgorithm, self).__init__(*args, **kwargs)

        # # replace the replay buffer with an ExpertReplayBuffer
        # # doing it like this i ugly but the easiest modification to make
        # self.replay_buffer = ExpertReplayBuffer(
        #     self.replay_buffer_size,
        #     self.observation_dim,
        #     self.action_dim,
        #     discrete_action_dim=self.discrete_action_dim,
        #     policy_uses_pixels=self.policy_uses_pixels
        # )

        # self.do_not_train = True


    def _do_training(self):
        pass
    

    @property
    def networks(self):
        return [self.exploration_policy]
