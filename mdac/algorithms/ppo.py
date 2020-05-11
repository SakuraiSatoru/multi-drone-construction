from torch.optim import Adam
from .net import LSTMPolicy2
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPO(object):

    def __init__(self, network_init_params,
                 gamma=0.99, lam=0.95, lr=0.001, coeff_entropy=5e-4, batch_size=1024):
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.coeff_entropy = coeff_entropy
        self.batch_size = batch_size
        self.pol_dev = 'cpu'  # device for policies
        self.niter = 0
        self.policy = LSTMPolicy2(**network_init_params)
        self.optim = Adam(self.policy.parameters(), lr=lr)

    @property
    def policies(self):
        return [self.policy]

    def step(self, observations, act_hidden, crt_hidden):
        out = [self.policy(obs, act_h, crt_h) for obs, act_h, crt_h in zip(observations, act_hidden, crt_hidden)]
        return list(map(list, zip(*out)))

    def update(self, buff, last_v, to_gpu=False):
        transformed_batches = self.transform_buffer(buff)
        for i, rollout in enumerate(transformed_batches):
            obs_batch, a_h_batch, a_c_batch, c_h_batch, c_c_batch, a_batch, r_batch, d_batch, l_batch, v_batch = rollout
            _last_v = np.asarray(last_v)[:, i, 0]
            t_batch, advs_batch = self.generate_train_data(rewards=r_batch,
                                                           gamma=self.gamma,
                                                           values=v_batch,
                                                           last_value=_last_v,
                                                           dones=d_batch,
                                                           lam=self.lam)
            memory = (obs_batch, a_h_batch, a_c_batch, c_h_batch, c_c_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)

            self._update(policy=self.policy, optimizer=self.optim,
                       batch_size=self.batch_size, memory=memory,
                       epoch=2, coeff_entropy=self.coeff_entropy,
                       clip_value=0.1)  # TODO hardcoded

    def prep_training(self, device='gpu'):
        self.policy.train()
        # self.critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.policy = fn(self.policy)
            self.pol_dev = device

    def prep_rollouts(self, device='cpu'):
        self.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            self.policy = fn(self.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {
                     'init_dict': self.init_dict,
                     'model_params':
                         {'policy': self.policy.state_dict(),
                          'optimizer': self.optim.state_dict()}
                     }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.99, lam=0.95, lr=0.001, coeff_entropy=5e-4, batch_size=1024):
        """
        Instantiate instance of this class from multi-agent environment
        """
        acsp = env.action_space[0]
        network_init_params = {
            'frames': 3,  # TODO hard coded
            'out_dim': acsp.shape[0]
        }
        init_dict = {
                     'gamma': gamma, 'lam': lam, 'lr': lr,
                     'coeff_entropy': coeff_entropy,
                     'batch_size': batch_size,
                     'network_init_params': network_init_params,
                     }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        if 'nagents' in save_dict['init_dict']:
            del(save_dict['init_dict']['nagents'])
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.policy.load_state_dict(save_dict['model_params']['policy'])
        instance.optim.load_state_dict(save_dict['model_params']['optimizer'])
        return instance

    # https://github.com/Acmece/rl-collision-avoidance/blob/40bf4f22b4270074d549461ea56ca2490b2e5b1c/model/ppo.py#L22
    def transform_buffer(self, buff):
        obs_batch, a_batch, r_batch, d_batch, l_batch, v_batch = [], [], [], [], [], []
        out = []
        a_h_batch = []
        a_c_batch = []
        c_h_batch = []
        c_c_batch = []
        for i in range(buff[0][0].shape[0]):  # rollout
            for e in buff:
                obs_batch.append(e[0][i])
                a_h_batch.append([h[i] for h, c in e[1]])
                a_c_batch.append([c[i] for h, c in e[1]])
                c_h_batch.append([h[i] for h, c in e[2]])
                c_c_batch.append([c[i] for h, c in e[2]])
                a_batch.append([a[i] for a in e[3]])
                r_batch.append(e[4][i])
                d_batch.append(e[5][i])
                assert e[6][0].shape[1] == 1
                l_batch.append([a[i, 0] for a in e[6]])
                assert e[7][0].shape[1] == 1
                v_batch.append([a[i, 0] for a in e[7]])

            out.append((np.asarray(obs_batch),
                        np.asarray(a_h_batch),
                        np.asarray(a_c_batch),
                        np.asarray(c_h_batch),
                        np.asarray(c_c_batch),
                        np.asarray(a_batch),
                        np.asarray(r_batch),
                        np.asarray(d_batch),
                        np.asarray(l_batch),
                        np.asarray(v_batch)))
        return out


    # https://github.com/Acmece/rl-collision-avoidance/blob/40bf4f22b4270074d549461ea56ca2490b2e5b1c/model/ppo.py#L122
    def generate_train_data(self, rewards, gamma, values, last_value, dones, lam):
        num_step = rewards.shape[0]
        num_env = rewards.shape[1]
        values = np.concatenate((values, last_value[None, :]))

        targets = np.zeros((num_step, num_env))
        gae = np.zeros((num_env,))

        for t in range(num_step - 1, -1, -1):
            delta = rewards[t, :] + gamma * values[t + 1, :] * (
                    1 - dones[t, :]) - values[t, :]
            gae = delta + gamma * lam * (1 - dones[t, :]) * gae

            targets[t, :] = gae + values[t, :]

        advs = targets - values[:-1, :]
        return targets, advs

    # https://github.com/Acmece/rl-collision-avoidance/blob/40bf4f22b4270074d549461ea56ca2490b2e5b1c/model/ppo.py#L143
    def _update(self, policy, optimizer, batch_size, memory, epoch,
                   coeff_entropy=0.02, clip_value=0.2):
        obss, a_h_states, a_c_states, c_h_states, c_c_states, actions, logprobs, targets, values, rewards, advs = memory
        n = obss.shape[0] * obss.shape[1]
        advs = (advs - advs.mean()) / advs.std()

        obss = obss.reshape(n, -1)
        a_h_states = a_h_states.reshape(n, -1)
        a_c_states = a_c_states.reshape(n, -1)
        c_h_states = c_h_states.reshape(n, -1)
        c_c_states = c_c_states.reshape(n, -1)
        actions = actions.reshape(n, -1)
        logprobs = logprobs.reshape(n, 1)
        advs = advs.reshape(n, 1)
        targets = targets.reshape(n, 1)

        for update in range(epoch):
            sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))),
                                   batch_size=batch_size,
                                   drop_last=False)
            for i, index in enumerate(sampler):
                sampled_obs = Variable(
                    torch.from_numpy(obss[index])).float().cuda()

                sampled_a_h_states = Variable(
                    torch.from_numpy(a_h_states[index])).float().cuda()
                sampled_a_c_states = Variable(
                    torch.from_numpy(a_c_states[index])).float().cuda()
                sampled_c_h_states = Variable(
                    torch.from_numpy(c_h_states[index])).float().cuda()
                sampled_c_c_states = Variable(
                    torch.from_numpy(c_c_states[index])).float().cuda()
                sampled_actions = Variable(
                    torch.from_numpy(actions[index])).float().cuda()
                sampled_logprobs = Variable(
                    torch.from_numpy(logprobs[index])).float().cuda()
                sampled_targets = Variable(
                    torch.from_numpy(targets[index])).float().cuda()
                sampled_advs = Variable(
                    torch.from_numpy(advs[index])).float().cuda()

                new_value, new_logprob, dist_entropy = policy.evaluate_actions(
                    sampled_obs, [sampled_a_h_states, sampled_a_c_states],
                    [sampled_c_h_states, sampled_c_c_states], sampled_actions)

                sampled_logprobs = sampled_logprobs.view(-1, 1)
                ratio = torch.exp(new_logprob - sampled_logprobs)

                sampled_advs = sampled_advs.view(-1, 1)
                surrogate1 = ratio * sampled_advs
                surrogate2 = torch.clamp(ratio, 1 - clip_value,
                                         1 + clip_value) * sampled_advs
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                sampled_targets = sampled_targets.view(-1, 1)
                value_loss = F.mse_loss(new_value, sampled_targets)

                loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

