import os
import random
import time
import logging
from typing import List, Tuple

import numpy as np

from arguments import get_args
from storage import RolloutStorage
import model1 as model
from env import DEFAULT_CAP_VALUE, NCOL, NROW, DecapPlaceParallel
from ppo import PPO


# Sets CUDA_VISIBLE_DEVICES BEFORE importing PyTorch
args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
import torch


torch.set_num_threads(1)

if __name__ == '__main__':
    now_time = time.strftime("%Y%m%d-%H%M", time.localtime(time.time()))
    t1 = ''.join([x for x in now_time if x.isdigit()])
    path = 'runs/case%s/' % (args.case_idx) + str(t1) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # args.learning_rate = 1e-4
    num_updates = 600

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # logging
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(path + 'log.txt')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # load env
    vec_env = DecapPlaceParallel([args.case_idx]*args.num_envs)
    actor_critic = model.PPONetwork(vec_env).to(device)
    # actor_critic.load_state_dict(torch.load('runs/case1/202409101544/vec_agent_params600.pth'))
    agent = PPO(actor_critic, args)
    logging.info("===========================================")
    logging.info("Environment: {}, Parallel Number: {}" .format(args.case_idx, args.num_envs))
    logging.info(f'Using GPU: {device}, {args.GPU}')
    logging.info(f'Learning Rate: {args.learning_rate:.2e}')
    logging.info(f'LR Schedule: {args.lr_schedule}')
    if args.lr_schedule == "exponential":
        logging.info(f'LR Anneal Factor: {args.anneal_lr_value}')
    elif args.lr_schedule == "step":
        logging.info(f'LR Step Size: {args.lr_step_size}, LR Gamma: {args.lr_gamma}')
    elif args.lr_schedule == "cosine":
        logging.info(f'LR Min: {args.lr_min}')
    elif args.lr_schedule == "cosine_warmup":
        logging.info(f'LR Step Size: {args.lr_step_size}, LR Cycle Mult: {args.lr_cycle_mult}, LR Min: {args.lr_min}')
    elif args.lr_schedule == "onecycle":
        logging.info(f'LR Max: {args.lr_max}')
    elif args.lr_schedule == "plateau":
        logging.info(f'LR Patience: {args.lr_patience}, LR Factor: {args.lr_factor}, LR Min: {args.lr_min}')
    elif args.lr_schedule == "cyclic":
        logging.info(f'LR Max: {args.lr_max}, LR Step Size: {args.lr_step_size}')
    elif args.lr_schedule == "restart":
        logging.info(f'LR Step Size: {args.lr_step_size}, LR Cycle Mult: {args.lr_cycle_mult}, LR Min: {args.lr_min}')
    elif args.lr_schedule == "linear":
        logging.info(f'LR Warmup Steps: {args.lr_warmup_steps}, LR Min: {args.lr_min}')
    logging.info("================ Training =================")


    # ALGO Logic: Storage setup
    rollouts = RolloutStorage(args.num_steps,
                              args.num_envs,
                              vec_env.SINGLE_OBSERVATION_SPACE_SHAPE,
                              vec_env.ACTION_SPACE_SHAPE,
                              vec_env.ACTION_SPACE)
    rollouts.to(device)

    loss = np.zeros(num_updates)
    pg_loss = np.zeros(num_updates)
    entropy_loss = np.zeros(num_updates)
    v_loss = np.zeros(num_updates)
    rewards = np.zeros((num_updates, args.num_steps * args.num_envs))
    learning_rates = np.zeros(num_updates)  # Track learning rates
    BEST = [-50, 0, 0]  # reward, updates, steps
    BEST_Allocation = np.zeros([])

    # Pre-allocate tensors to avoid repeated allocations
    temp_obs = torch.zeros((args.num_envs,) + vec_env.SINGLE_OBSERVATION_SPACE_SHAPE, device=device)
    temp_imped = torch.zeros((args.num_envs, 231*4), device=device)
    temp_action_mask = torch.zeros((args.num_envs, vec_env.ACTION_LOCATIONS), device=device)
    temp_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    temp_reward = torch.zeros(args.num_envs, device=device)
    
    start_time = time.time()
    # setup the initial best decap values
    BEST_DecapValues = DEFAULT_CAP_VALUE*NCOL*NROW
    
    for update in range(1, num_updates + 1):
        vec_obs, vec_imped = vec_env.reset()
        # Efficient tensor conversion with pre-allocated tensors
        temp_obs.copy_(torch.from_numpy(vec_obs))
        temp_imped.copy_(torch.from_numpy(vec_imped))
        temp_action_mask.copy_(torch.from_numpy(np.stack(vec_env.vec_action_mask())))
        
        rollouts.obs[0].copy_(temp_obs)
        rollouts.imped[0].copy_(temp_imped)
        rollouts.action_masks[0].copy_(temp_action_mask)
        
        next_obs, next_imped, vec_action_mask = temp_obs, temp_imped, temp_action_mask

        for step in range(0, args.num_steps):

            #############################################################
            ###### Obs + Action Masks => Agent => Actions################
            ###### Actions => Env => Rewards, log_probs, New Action Maks#

            with torch.no_grad():
                vec_action, vec_logprob, _, vec_value = actor_critic.get_action_and_value(next_obs, next_imped, vec_action_mask)

            # Cal rewards for given actions
            vec_obs, vec_imped, vec_reward, vec_done, info = vec_env.step(vec_action.cpu().numpy())

            #############################################################
            #############################################################

            # Efficient tensor updates using pre-allocated tensors
            temp_done.copy_(torch.from_numpy(np.array(vec_done)))
            temp_obs.copy_(torch.from_numpy(vec_obs))
            temp_imped.copy_(torch.from_numpy(vec_imped))
            temp_reward.copy_(torch.from_numpy(np.array(vec_reward)))
            
            next_done, next_obs, next_imped = temp_done, temp_obs, temp_imped

            # Optimize: Calculate max only once and reuse
            max_reward = max(info["reward_now"])
            if max_reward > BEST[0]:
                BEST = max_reward, update, step
                index = info["reward_now"].index(max_reward)  # More efficient than enumerate+max
                BEST_Allocation = vec_env.vec_cur_params_idx[index]

            # When the target impedance is satisfied or there is no valid positions, Done is True
            # The final reward is propagated to the before states.
            reset_indices = []
            for idx in range(vec_env.env_count):
                if vec_reward[idx] < info["his_reward"][idx]:
                    temp_reward[idx] = vec_reward[idx] - 0.1 # penalty for reward decrease
                if info["reward_now"][idx] > 0 or sum(vec_action_mask[idx]) == 0:
                    next_done[idx] = True
                    reset_indices.append(idx)
                    # update BEST_DecapValues for reward > 0
                    if BEST_DecapValues > vec_env.vec_cur_params_idx[idx].sum():
                        BEST_DecapValues = vec_env.vec_cur_params_idx[idx].sum()
                elif vec_env.vec_cur_params_idx[idx].sum() >= BEST_DecapValues:
                    temp_reward[idx] = vec_reward[idx] - 0.1 # penalty for Decapvalues 
                    next_done[idx] = True
                    reset_indices.append(idx)
            
            # Batch process environment resets to reduce overhead
            if reset_indices:
                for idx in reset_indices:
                    obs_done, imped_done = vec_env.reset_idx(idx)
                    temp_obs[idx].copy_(torch.from_numpy(obs_done))
                    temp_imped[idx].copy_(torch.from_numpy(imped_done))

            temp_action_mask.copy_(torch.from_numpy(np.stack(vec_env.vec_action_mask())))
            vec_action_mask = temp_action_mask

            rollouts.insert(step, next_obs, next_imped, vec_action.reshape([-1, vec_env.ACTION_SPACE_SHAPE[0]]),
                            vec_logprob, temp_reward.view(-1), next_done, vec_value.flatten(),
                            vec_action_mask)

        with torch.no_grad():
            next_value = actor_critic.get_value(next_obs, next_imped).reshape(1, -1)
            rollouts.advantages, rollouts.returns = rollouts.compute_returns(args.num_steps, args.gae, next_value, next_done,
                                                             args.gamma, args.gae_lambda,
                                                             rollouts.values, rollouts.rewards, rollouts.dones)

        v_loss[update - 1], pg_loss[update - 1], entropy_loss[update - 1], loss[update - 1] = agent.update(rollouts,
                                                                                                           args.num_steps,
                                                                                                           vec_env.SINGLE_OBSERVATION_SPACE_SHAPE,
                                                                                                           vec_env.ACTION_SPACE_SHAPE)
        
        # Step learning rate scheduler
        current_reward = max_reward if 'max_reward' in locals() else (max(info["reward_now"]) if info["reward_now"] else 0.0)
        current_lr = agent.step_lr_scheduler(update, current_reward)
        learning_rates[update - 1] = current_lr
        
        rewards[update - 1] = rollouts.rewards.cpu().numpy().reshape(-1)

        logging.info(f"-------------------- Update {update} --------------------")
        logging.info(f'Current Learning Rate: {current_lr:.2e}')
        logging.info('Best Update :{}, Best Step :{}, Best Reward: {}'.format( BEST[1], BEST[2], BEST[0]))
        np.savetxt(path + 'allocation.txt', BEST_Allocation)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('time cost: {} s'.format(total_time))

    # save network parameters
    torch.save(actor_critic, path + 'vec_agent.pth')
    torch.save(actor_critic.state_dict(), path + 'vec_agent_params.pth')

    # save data
    np.savetxt(path + 'reward.txt', rewards)
    np.savetxt(path + 'loss.txt', loss)
    np.savetxt(path + 'pgloss.txt', pg_loss)
    np.savetxt(path + 'entloss.txt', entropy_loss)
    np.savetxt(path + 'vloss.txt', v_loss)
    np.savetxt(path + 'learning_rates.txt', learning_rates)