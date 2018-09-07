if self.discrete_policy:
            q_pred = self.qf(obs)
            v_pred = self.vf(obs)
            # Make sure policy accounts for squashing functions like tanh correctly!
            log_pis = self.policy.get_log_pis(obs)

            """
            QF Loss
            """
            target_v_values = self.target_vf(next_obs)
            q_target = rewards + (1. - terminals) * self.discount * target_v_values
            qf_loss = 0.5 * torch.mean(
                (torch.sum(q_pred*actions, 1) - q_target.detach())**2
            )
            print('-----')
            print(rewards[0].data.numpy())
            print(q_target[0].data.numpy())
            print(q_pred[0].data.numpy())
            print(actions[0].data.numpy())
            print(log_pis[0].data.numpy())

            """
            VF Loss
            """
            # print('-----')
            # print(torch.exp(log_pis)[0])
            # print(F.softmax(q_pred, 1)[0])
            v_target = torch.sum(torch.exp(log_pis) * (q_pred - log_pis), 1).detach()
            vf_loss = 0.5 * torch.mean((v_pred - v_target)**2)
            # print('------')
            # print(v_target[0].data.numpy())
            # print(q_pred[0].data.numpy())
            # print(log_pis[0].data.numpy())

            """
            Policy Loss
            """
            policy_loss = self.nn_KL(log_pis, F.softmax(q_pred, dim=1).detach()) / obs.size(0)