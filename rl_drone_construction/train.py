import os
import numpy as np
import torch
from pathlib import Path
from make_env import make_env
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from algorithms.ppo import PPO
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv

ENV_ID = 'trainstage1'
RANDOM_SEED = 123
N_EPISODES = 50000
N_ROLLOUT_THREADS = 1
EPISODE_LEN = 80
SAVE_INTERVAL = 500

USE_CUDA = torch.cuda.is_available()

GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
LR = 2e-5  # 5e-5 stage 1


def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def train():

    model_dir = Path('./rl_drone_construction/models') / ENV_ID
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    logger = SummaryWriter(str(log_dir), max_queue=5, flush_secs=30)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env = make_parallel_env(ENV_ID, N_ROLLOUT_THREADS, RANDOM_SEED)
    ppo = PPO.init_from_env(env, gamma=GAMMA, lam=LAMDA, lr=LR,
                            coeff_entropy=COEFF_ENTROPY,
                            batch_size=BATCH_SIZE)

    # save_dict = torch.load(model_dir / 'run1/model.pt')  # TODO init from save?
    # save_dict = save_dict['model_params']['policy']
    # ppo.policy.load_state_dict(save_dict)

    buff = []

    t = 0
    for ep_i in range(0, N_EPISODES, N_ROLLOUT_THREADS):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + N_ROLLOUT_THREADS,
                                        N_EPISODES))
        obs = env.reset()
        nagents = obs.shape[1]
        ppo.prep_rollouts(device='cpu')

        ep_rew = 0
        act_hidden = [[torch.zeros(N_ROLLOUT_THREADS, 128), torch.zeros(N_ROLLOUT_THREADS, 128)] for i in range(nagents)]
        crt_hidden = [[torch.zeros(N_ROLLOUT_THREADS, 128), torch.zeros(N_ROLLOUT_THREADS, 128)] for i in range(nagents)]

        for et_i in range(EPISODE_LEN):

            """
            generate actions
            """
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(nagents)]
            prev_act_hidden = [[h.data.cpu().numpy(), c.data.cpu().numpy()] for h, c in act_hidden]
            prev_crt_hidden = [[h.data.cpu().numpy(), c.data.cpu().numpy()] for h, c in crt_hidden]
            v_list, agent_actions_list, logprob_list, mean_list = ppo.step(torch_obs, act_hidden, crt_hidden)
            v_list = [a.data.cpu().numpy() for a in v_list]
            agent_actions_list = [a.data.cpu().numpy() for a in agent_actions_list]
            logprob_list = [a.data.cpu().numpy() for a in logprob_list]
            clipped_action_list = [np.clip(a, -1, 1) for a in agent_actions_list]
            actions = [[ac[i] for ac in clipped_action_list] for i in
                       range(N_ROLLOUT_THREADS)]

            """
            step env
            """
            next_obs, rewards, dones, infos = env.step(actions)
            ep_rew += np.mean(rewards)
            buff.append((obs, prev_act_hidden, prev_crt_hidden, agent_actions_list, rewards, dones, logprob_list, v_list))
            obs = next_obs
            t += N_ROLLOUT_THREADS

            for i, done in enumerate(dones[0]):
                if done:
                    act_hidden[i] = [torch.zeros(N_ROLLOUT_THREADS, 128), torch.zeros(N_ROLLOUT_THREADS, 128)]
                    crt_hidden[i] = [torch.zeros(N_ROLLOUT_THREADS, 128), torch.zeros(N_ROLLOUT_THREADS, 128)]
                    env.envs[0].agents[i].terminate = False
            # if dones.any():
            #     break
        print('mean reward:', ep_rew)

        """
        train
        """
        next_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(nagents)]
        v_list, _, _, _ = ppo.step(next_obs, act_hidden, crt_hidden)
        v_list = [a.data.cpu().numpy() for a in v_list]

        print('updating params...')
        if USE_CUDA:
            ppo.prep_training(device='gpu')
        else:
            ppo.prep_training(device='cpu')
        ppo.update(buff=buff, last_v=v_list, to_gpu=USE_CUDA)
        ppo.prep_rollouts(device='cpu')
        buff = []

        logger.add_scalar('mean_episode_rewards', ep_rew, ep_i)

        if ep_i % SAVE_INTERVAL < N_ROLLOUT_THREADS:
            print('saving incremental...')
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            ppo.save(
                str(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1))))
            ppo.save(str(run_dir / 'model.pt'))

    print('saving model...')
    ppo.save(str(run_dir / 'model.pt'))
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    train()
