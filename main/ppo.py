import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO():
    def __init__(self, actor_critic, args):

        self.actor_critic = actor_critic
        self.clip_coef = args.clip_coef
        self.update_epochs = args.update_epochs
        self.batch_size = args.batch_size
        self.minibatch_size = args.minibatch_size
        self.norm_adv = args.norm_adv
        self.clip_vloss = args.clip_vloss
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.target_kl = args.target_kl

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=args.learning_rate, eps=1e-5)
        self.initial_lr = args.learning_rate
        self.lr_schedule = args.lr_schedule
        self.lr_step_size = args.lr_step_size
        self.lr_gamma = args.lr_gamma
        self.lr_warmup_steps = args.lr_warmup_steps
        self.lr_min = args.lr_min
        self.lr_max = args.lr_max
        self.lr_patience = args.lr_patience
        self.lr_factor = args.lr_factor
        self.lr_cycles = args.lr_cycles
        self.lr_cycle_mult = args.lr_cycle_mult
        self.current_step = 0
        self.best_reward = float('-inf')
        self.plateau_counter = 0
        
        # Initialize learning rate scheduler based on strategy
        if args.lr_schedule == "exponential":
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.anneal_lr_value)
        elif args.lr_schedule == "step":
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_schedule == "cosine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=args.lr_min)
        elif args.lr_schedule == "cosine_warmup":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=args.lr_step_size, T_mult=args.lr_cycle_mult, eta_min=args.lr_min)
        elif args.lr_schedule == "onecycle":
            self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=args.lr_max, total_steps=1000, 
                pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e4)
        elif args.lr_schedule == "plateau":
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience, 
                min_lr=args.lr_min, verbose=True)
        elif args.lr_schedule == "cyclic":
            self.lr_scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=args.lr_min, max_lr=args.lr_max, 
                step_size_up=args.lr_step_size, step_size_down=args.lr_step_size,
                mode='triangular', cycle_momentum=False)
        elif args.lr_schedule == "restart":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=args.lr_step_size, T_mult=args.lr_cycle_mult, eta_min=args.lr_min)
        elif args.lr_schedule == "linear":
            self.lr_scheduler = None  # Custom linear scheduler
        else:
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.anneal_lr_value)

    def update(self, rollouts, num_steps, obs_shape, action_shape):

        # flatten the batch
        b_obs = rollouts.obs[0:num_steps].reshape((-1,) + obs_shape)
        b_imped = rollouts.imped[0:num_steps].reshape((-1, rollouts.imped.shape[-1]))
        b_logprobs = rollouts.logprobs.reshape(-1)
        b_actions = rollouts.actions.reshape((-1,) + action_shape)
        b_advantages = rollouts.advantages.reshape(-1,)
        b_returns = rollouts.returns.reshape(-1,)
        b_values = rollouts.values.reshape(-1,)
        b_action_masks = rollouts.action_masks[0:num_steps].reshape((-1, rollouts.action_masks.shape[-1]))

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        v_loss_epoch = 0
        pg_loss_epoch = 0
        entropy_loss_epoch = 0
        loss_epoch = 0
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = self.actor_critic.get_action_and_value(b_obs[mb_inds],
                                                                                          b_imped[mb_inds],
                                                                                          b_action_masks[mb_inds],
                                                                                          b_actions.long()[mb_inds].T)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()


                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.clip_coef,
                                                                self.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()
                # loss = policy_loss - entropy * entropy_coefficient + value_loss * value_coefficient
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                v_loss_epoch += v_loss.item()
                pg_loss_epoch += pg_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                loss_epoch += loss.item()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

        num_updates = self.update_epochs * int(self.batch_size/self.minibatch_size)

        v_loss_epoch /= num_updates
        pg_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates
        loss_epoch /= num_updates

        return v_loss_epoch, pg_loss_epoch, entropy_loss_epoch, loss_epoch
    
    def step_lr_scheduler(self, update_step: int, current_reward: float = None) -> float:
        """Step the learning rate scheduler and return current learning rate."""
        self.current_step = update_step
        
        if self.lr_schedule == "linear":
            # Custom linear decay
            if update_step < self.lr_warmup_steps:
                # Warmup phase: linearly increase from 0 to initial_lr
                lr = self.initial_lr * (update_step / self.lr_warmup_steps)
            else:
                # Linear decay phase
                progress = (update_step - self.lr_warmup_steps) / max(1, 1000 - self.lr_warmup_steps)
                lr = self.initial_lr * (1.0 - progress * 0.9)  # Decay to 10% of initial LR
                lr = max(lr, self.lr_min)
            
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
        elif self.lr_schedule == "plateau":
            # Plateau scheduler needs reward information
            if current_reward is not None:
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    self.plateau_counter = 0
                else:
                    self.plateau_counter += 1
                
                # Step the scheduler with the reward metric
                self.lr_scheduler.step(current_reward)
            lr = self.optimizer.param_groups[0]['lr']
            
        elif self.lr_schedule == "onecycle":
            # OneCycle scheduler steps every batch, so we step it here
            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
        elif self.lr_scheduler is not None:
            # Use PyTorch scheduler
            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
        else:
            lr = self.optimizer.param_groups[0]['lr']
        
        return lr
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
