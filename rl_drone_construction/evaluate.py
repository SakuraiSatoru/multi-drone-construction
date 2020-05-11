import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from make_env import make_env
from algorithms.ppo import PPO

def run(config):
    model_path = (Path('./rl_drone_construction/models') / config.env_id /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    ppo = PPO.init_from_save(model_path)
    env = make_env(config.env_id)
    ppo.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        nagents = len(obs)

        dones = [False] * nagents
        for agent in env.agents:
            agent.trajectory = []

        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        act_hidden = [[torch.zeros(1, 128), torch.zeros(1, 128)] for i in range(nagents)]
        crt_hidden = [[torch.zeros(1, 128), torch.zeros(1, 128)] for i in range(nagents)]
        rews = None

        env.render('human')

        for t_i in range(config.episode_length):
            print(f'{t_i} / {config.episode_length}')
            calc_start = time.time()
            nagents = len(obs)
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(nagents)]
            _, _, _, mean_list = ppo.step(torch_obs, act_hidden, crt_hidden)
            agent_actions_list = [a.data.cpu().numpy() for a in mean_list]
            clipped_action_list = [np.clip(a, -1, 1) for a in agent_actions_list]
            actions = [ac.flatten() for ac in clipped_action_list]
            for i in range(len(dones)):
                if dones[i]:
                    env.agents[i].movable = False
                else:
                    env.agents[i].trajectory.append(np.copy(env.agents[i].state.p_pos))
            obs, rewards, dones, infos = env.step(actions)
            if rews is None:
                rews = np.zeros(len(rewards))
            rews += np.array(rewards)
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("run_num", type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=120, type=int)
    parser.add_argument("--fps", default=40, type=int)

    config = parser.parse_args()

    run(config)
