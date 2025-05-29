import gymnasium as gym
from custom_envs.environment import  SimpleGridEnv, DynamicGridEnv
import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical
from NeuralNet import ActorCritic, ReplayBuffer
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PPO:
    def __init__(self, env_name, lr, gamma, lam, minibatch_size, num_epochs, eps_clip, target_KL = None):
        self.env = gym.make(env_name, num_obstacles = 3, nrow = 5, ncol= 5)
        self.state_dim  = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.num_epochs = num_epochs
        self.batch_size = minibatch_size
        self.target_KL = target_KL
        self.memory = ReplayBuffer()
        self.policy = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.lr, eps=1e-5)
        self.policy_old = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.Loss = nn.MSELoss()
        self.reward_history = []

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, logprob, state_val = self.policy_old.act(state)
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(logprob)
        self.memory.state_values.append(state_val)
        return action.item()


    def compute_gae(self, rewards, values, dones, next_state):
        
        with torch.no_grad():
            advantages = []
            gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards)-1:
                    next_value = self.policy.critic(next_state)
                else:
                    next_value = values[t+1]
                
                #dones[t] IS REFERRING TO TERMINAL CONDITION OF THE NEXT STATE = values[t+1]
                mask = 1-dones[t]
                delta = rewards[t] + self.gamma * mask * next_value  - values[t]
                gae = delta + self.gamma * self.lam * mask * gae
                advantages.insert(0, gae)



                #mask = 1-dones[t]
                #next_value = mask * next_value
                #delta = rewards[t] + self.gamma * next_value  - values[t]
                #gae = delta + self.gamma * self.lam * mask * gae
                #advantages.insert(0, gae)
                #next_value = values[t]

        return torch.tensor(advantages, dtype=torch.float32)

    def update_gae(self, next_state_gae):
        rewards      = torch.tensor(self.memory.rewards, dtype=torch.float32).to(device)
        dones        = torch.tensor(self.memory.is_terminals, dtype=torch.float32).to(device)
        old_states   = torch.stack(self.memory.states).to(device).detach()
        old_actions  = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        values       = torch.tensor(self.memory.state_values, dtype=torch.float32).to(device)
        advantages = self.compute_gae(rewards, values, dones, next_state_gae)
        returns = advantages + values
        clipfracs = []
        
        for _ in range(self.num_epochs):    
            indices = np.arange(len(old_states))
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):

                batch_indices = indices[i: i + self.batch_size]
                states_batch = old_states[batch_indices]
                actions_batch = old_actions[batch_indices]
                logprobs_batch = old_logprobs[batch_indices]
                advantages_batch = advantages[batch_indices]
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)


                logprobs, state_values, dist_entropy = self.policy.evaluate(states_batch, actions_batch)
                ratios = torch.exp(logprobs - logprobs_batch) #.detach()
                
                with torch.no_grad():
                    approx_kl = ((ratios - 1) - logprobs + logprobs_batch).mean()
                    clipfracs += [((ratios - 1.0).abs() > self.eps_clip).float().mean().item()]

                surrogate_obj = ratios * advantages_batch
                clipped_surrogate_obj = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_batch

                critic_loss = 0.5 * self.Loss(state_values.squeeze(), returns[batch_indices]) #rewards
                loss = -torch.min(surrogate_obj, clipped_surrogate_obj).mean() + critic_loss - 0.01 * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

            if self.target_KL is not None:
                if approx_kl > self.target_KL:
                    print('KULBACK LEIBLER GREATER THAN TARGET')
                    break
            

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only = True))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only = True))


    def train(self, train_step, update_timestep):
        
        timesteps = 0
        num_updates = 0
        results_dir = "Autonomous_Projects/Results"
        os.makedirs(results_dir, exist_ok=True)
        state = self.env.reset(options = options)[0]
        episode_reward = 0
        episode = 0
        total_updates = train_step/update_timestep
        for _ in range(train_step):
            
            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.memory.rewards.append(reward)
            self.memory.is_terminals.append(done)
            episode_reward += reward
            state = next_state
            timesteps += 1
            
            if timesteps % update_timestep == 0:
                    ###HERE UPDATE THE LERNING RATE USING LR ANNEALING
                    next_gae_state = torch.tensor(state, dtype= torch.float32)
                    num_updates += 1
                    frac = 1 - (num_updates-1)/total_updates
                    new_lr = self.lr*frac
                    new_lr = max(new_lr, 0)
                    self.optimizer.param_groups[0]["lr"] = new_lr
                    self.update_gae(next_gae_state)

            ##IF DONE THE EPISODE IS TERMINATED AND SO UPDATE THE STATS
            if done:
                
                episode += 1
                self.reward_history.append(episode_reward)
                print(f"Episode {episode}, Reward: {episode_reward}")
                episode_reward = 0
                #If episode terminates reset the environment
                state = self.env.reset(options = options)[0]

        #At the end of training iterations save the model parameters
        self.save("Autonomous_Projects/maze.pth")
        


    def plot_train(self):

        plt.figure()
        plt.plot(self.reward_history)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("PPO Rewards")
        plt.savefig("Autonomous_Projects/Results/plot_training.png")
        plt.close()
        self.env.close()

    def test(self, checkpoint_path, episodes=10):
        self.load(checkpoint_path)
        test_env = gym.make(env_name, num_obstacles = 3, nrow = 5, ncol= 5, render_mode = 'human')
        for episode in range(episodes):
            state = test_env.reset(options = options)[0]
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, done, _, _ = test_env.step(action)
                total_reward += reward
            print(f"Test Episode {episode + 1}: Reward = {total_reward}")    
    
    def record(self, checkpoint_path, episodes = 3):
        self.load(checkpoint_path)
        record_env = gym.make("MovingObstaclesGrid-v0", num_obstacles = 3, nrow = 5, ncol= 5, render_mode='rgb_array')
        record_env = gym.wrappers.RecordVideo(record_env, video_folder="videos", episode_trigger=lambda x: True, name_prefix= "MovingObstacles")
        for episode in range(episodes):
            state = record_env.reset(options = options)[0]
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, done, _, _ = record_env.step(action)
                total_reward += reward
            print(f"Test Episode {episode + 1}: Reward = {total_reward}")
        record_env.close()
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="train or test the agent")
    args = parser.parse_args()
    
    env_name = 'MovingObstaclesGrid-v0'
    options ={
        'goal_loc' : (3,3)
        }
    ppo = PPO(env_name, lr = 1e-3, gamma = 0.99, lam = 0.95, minibatch_size= 32, num_epochs = 10, eps_clip = 0.2, target_KL=0.01)
    if args.mode == "train":
        ppo.train(train_step= 100000, update_timestep = 2000)
        ppo.plot_train()
    elif args.mode == "test":
        ppo.test("Autonomous_Projects/maze.pth")
    elif args.mode == "record":
        ppo.record("Autonomous_Projects/maze.pth")
