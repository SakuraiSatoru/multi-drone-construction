import argparse
import torch
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from make_env import make_env
from rl_drone_construction.algorithms.ppo import PPO
from utils.ws_server import WebSocket


class Simulation(object):
    def __init__(self, config):
        print("initializing env...", end="")
        model_path = Path('./rl_drone_construction/models/trainstage2/run1/model.pt')
        if config.save:
            self.stream_path = Path('./vis/stream') / config.save
            if os.path.exists(self.stream_path):
                raise FileExistsError
        else:
            self.stream_path = None
        if config.load:
            self.load_path = Path('./vis/stream') / config.load
            self.fp = open(self.load_path, 'r')
            self.prev_json = None
        else:
            self.load_path = None
        self.ppo = PPO.init_from_save(model_path)
        self.env = make_env(config.env_id)
        self.ppo.prep_rollouts(device='cpu')
        self.obs = self.env.reset()
        self.nagents = len(self.obs)
        # self.env.render('human')
        self.act_hidden = [[torch.zeros(1, 128), torch.zeros(1, 128)] for i
                              in range(self.nagents)]
        self.crt_hidden = [[torch.zeros(1, 128), torch.zeros(1, 128)] for i
                              in range(self.nagents)]
        print("done")

    def step(self):
        if self.load_path:
            j = self.fp.readline()
            if not j:
                return self.prev_json
            self.prev_json = j
            return j

        # rearrange observations to be per agent, and convert to torch Variable
        self.nagents = len(self.obs)
        while len(self.act_hidden) > self.nagents:
            self.act_hidden.pop()
            self.crt_hidden.pop()
            self.env.action_space.pop()
            self.env.observation_space.pop()
        while len(self.act_hidden) < self.nagents:
            self.env.action_space.append(self.env.action_space[0])
            self.env.observation_space.append(self.env.observation_space[0])
            self.act_hidden.append(
                [torch.zeros(1, 128), torch.zeros(1, 128)])
            self.crt_hidden.append(
                [torch.zeros(1, 128), torch.zeros(1, 128)])
        torch_obs = [Variable(torch.Tensor(self.obs[i]).view(1, -1),
                              requires_grad=False) for i in
                     range(self.nagents)]
        _, _, _, mean_list = self.ppo.step(torch_obs, self.act_hidden, self.crt_hidden)
        agent_actions_list = [a.data.cpu().numpy() for a in mean_list]
        clipped_action_list = [np.clip(a, -1, 1) for a in agent_actions_list]
        actions = [ac.flatten() for ac in clipped_action_list]

        self.obs, rewards, dones, infos = self.env.step(actions)
        # self.env.render('human')

        d = self.env.world.gather_stream_data()
        if self.stream_path:
            with open(self.stream_path, 'a') as fp:
                fp.write(d + '\n')
        return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("--save", default=None, type=str,
                        help="Save ws packets to a file for future use")
    parser.add_argument("--load", default=None, type=str,
                        help="Load saved ws packets file to avoid computing")

    config = parser.parse_args()

    s = Simulation(config)
    ws = WebSocket(producer=s.step)
    ws.run()
