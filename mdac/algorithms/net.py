import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


class LSTMPolicy2(nn.Module):
    def __init__(self, frames, out_dim):
        super(LSTMPolicy2, self).__init__()
        self.frames = frames
        self.logstd = nn.Parameter(torch.zeros(out_dim))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(64*32, 256)  # TODO hard coded
        self.act_fc2 = nn.Linear(256+2+2, 128)  # TODO hard coded
        self.act_lstm = nn.LSTMCell(input_size=128, hidden_size=128)
        self.actor = nn.Linear(128, 2)

        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(64*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.crt_lstm = nn.LSTMCell(input_size=128, hidden_size=128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, act_hidden, crt_hidden):
        """
            returns value estimation, action, log_action_prob
        """
        in_dim = x.shape[1]
        lidar = x[:, :in_dim-4].view(x.shape[0], self.frames, -1)  # TODO hard coded
        others = x[:, in_dim-4:]
        # action
        a = F.relu(self.act_fea_cv1(lidar))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))
        a = torch.cat((a, others), dim=-1)
        a = F.relu(self.act_fc2(a))
        act_hidden[0], act_hidden[1] = self.act_lstm(a, act_hidden)
        mean = torch.tanh(self.actor(act_hidden[0]))

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = self.log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        v = F.relu(self.crt_fea_cv1(lidar))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, others), dim=-1)
        v = F.relu(self.crt_fc2(v))
        crt_hidden[0], crt_hidden[1] = self.crt_lstm(v, crt_hidden)
        v = self.critic(v)

        return v, action, logprob, mean

    # https://github.com/Acmece/rl-collision-avoidance/blob/40bf4f22b4270074d549461ea56ca2490b2e5b1c/model/net.py#L72
    def evaluate_actions(self, x, act_hidden, crt_hidden, action):
        v, _, _, mean = self.forward(x, act_hidden, crt_hidden)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = self.log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

    # https://github.com/Acmece/rl-collision-avoidance/blob/40bf4f22b4270074d549461ea56ca2490b2e5b1c/model/utils.py#L90
    def log_normal_density(self, x, mean, log_std, std):
        """returns guassian density given x on log scale"""

        variance = std.pow(2)
        log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * \
                      np.log(2 * np.pi) - log_std  # num_env * frames * act_size
        log_density = log_density.sum(dim=-1, keepdim=True)  # num_env * frames * 1
        return log_density



