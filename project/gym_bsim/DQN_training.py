import collections
import os
import random
import gymnasium
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym_bsim
import rl_utils
from utils import *
import numpy as np


def optimize_actions(buffer):
    actions = list(buffer)
    optimized_actions = []
    i = 0
    while i < len(actions):
        current = actions[i]
        if current % 2 == 0:
            target = current + 1
        else:
            target = current - 1

        if target in actions[i + 1:]:
            actions.remove(current)
            actions.remove(target)
        else:
            optimized_actions.append(current)
            i += 1

    return optimized_actions[:15]


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.reward_count = 0
        self.rms = None

    def add(self, action):
        self.buffer.append(action)

    def size(self):
        return len(self.buffer)

    def rms_record(self, rms):
        self.rms = rms

    def get_rms(self):
        return self.rms

    def clear(self):
        self.buffer = collections.deque(maxlen=self.capacity)

    def save_to_txt(self, filename):
        with open(filename, 'w') as f:
            for action in self.buffer:
                f.write(f"{action}\n")


class PriorBuffer(object):
    def __init__(self, buffer_size, parameter):
        self.ptr = 0
        self.size = 0

        max_size = int(buffer_size)
        self.state = np.zeros((max_size, max(param_data['index'] for param_data in parameter.values()) + 1))
        self.action = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, max(param_data['index'] for param_data in parameter.values()) + 1))
        self.done = np.zeros((max_size, 1))
        self.max_size = max_size

        self.sum_tree = SumTree(max_size)
        self.alpha = 0.6
        self.beta = 0.6

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        priority = 1.0 if self.size == 0 else self.sum_tree.priority_max
        self.sum_tree.update_priority(data_index=self.ptr, priority=priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind, Normed_IS_weight = self.sum_tree.prioritized_sample(N=self.size, batch_size=batch_size, beta=self.beta)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind],
            ind,
            Normed_IS_weight
        )

    def update_batch_priorities(self, batch_index, td_errors):
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update_priority(data_index=index, priority=priority)

    def len(self):
        return self.size


class VAnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, param_high, param_low):
        super(VAnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(128, 256)
        self.fc_A = nn.Linear(128, action_dim)
        self.fc_V = nn.Linear(128, 1)

        self.high = torch.tensor(param_high, dtype=torch.float)
        self.low = torch.tensor(param_low, dtype=torch.float)

    def forward(self, x):
        x = (x - self.low) / (self.high - self.low)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Normalized input contains NaN or Inf values")
        x = F.relu(self.fc1(x))
        A = self.fc_A(x)
        V = self.fc_V(x)
        Q = V + A - A.mean(1).view(-1, 1)
        return Q


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, param_high, param_low, args):
        self.action_dim = action_dim
        self.q_net = VAnet(state_dim, hidden_dim, self.action_dim, param_high, param_low).to(args.device)
        self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim, param_high, param_low).to(args.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.gamma = args.gamma
        self.final_epsilon = args.epsilon_final
        self.initial_epsilon = args.epsilon_initial
        self.epsilon1 = args.epsilon_initial
        self.target_update = args.target_update
        self.count = 0
        self.device = args.device
        self.losses = []
        self.epoch_losses = []
        self.q_values = 0
        self.q_values_sum = []

    def take_action(self, state):
        if np.random.random() < self.epsilon1:
            self.epsilon1 -= (self.initial_epsilon - self.final_epsilon) / 2000
            # print(f'current epsilon1: {self.epsilon1}')
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def take_action2(self):
        return np.random.randint(self.action_dim)

    def save_model(self, filepath):
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def update(self, transition_dict1, replay_buffer1):
        states = torch.tensor(transition_dict1['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict1['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict1['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict1['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict1['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        ind = torch.tensor(transition_dict1['ind'].astype(np.int64), dtype=torch.long)
        normed_is_weight = transition_dict1['Normed_IS_weight'].view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        self.q_values = torch.mean(q_values).item()

        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        td_errors = (q_values - q_targets).squeeze(-1)
        dqn_loss = torch.mean(normed_is_weight * (td_errors ** 2))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

        replay_buffer1.update_batch_priorities(ind, td_errors.detach().cpu().numpy())

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1
        self.losses.append(dqn_loss.item())

    def loss_record(self):
        self.epoch_losses.extend(self.losses)
        self.losses = []

    def q_value_record(self):
        self.q_values_sum.append(self.q_values)
        print(f'q value：{self.q_values}\n')


def train(args):
    processed_indices = set()
    first_state = []
    param_high = []
    param_low = []
    for param, param_info in args.parameter.items():
        param_index = param_info['index']
        if param_index not in processed_indices:
            first_state.append(param_info['initial'])
            param_high.append(param_info['high'])
            param_low.append(param_info['low'])
            processed_indices.add(param_index)
    first_state = np.array(first_state)
    param_high = np.array(param_high)
    # print(param_high)
    param_low = np.array(param_low)

    PriorBuffer1 = PriorBuffer(args.buffer_size, args.parameter)
    env = gymnasium.make('gym_bsim/bsim-v0', first_state=first_state, high=param_high, low=param_low,
                         t_limit=args.t_limit,
                         step_size=args.step_size, train_model=True, task_batch_path=args.task_batch_path,
                         target=args.target_path, fitting=args.fitting_path,
                         sp=args.sp_path, model=args.model_path, parameter=args.parameter, curve_type=args.curve_type)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    shortest_path_pool = ReplayBuffer(100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, args.hidden_dim, action_dim, param_high, param_low, args)

    pretraining_model_path = rf'{args.task_batch_path[0]}\results\pretraining_model\your_modelpth'
    if os.path.exists(pretraining_model_path):
        print(f"Loading existing model weights from {pretraining_model_path}")
        checkpoint = torch.load(model_path, map_location=args.device, weights_only=True)
        agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        agent.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model weights loaded successfully\n", '*'*100)
    else:
        print("No pretrained model found, training from scratch\n", '*'*100)

    return_list = []
    short_path_min_rms = 100

    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                print('-' * 100)
                episode_return = 0
                episode_buffer = []
                action_record = []
                shortest_path_len = len(shortest_path_pool.buffer)

                if shortest_path_len > 0:
                    if np.random.random() < args.epsilon:
                        state, reset_info = env.reset()
                        initial_state = True
                    else:
                        state, reset_info = env.reset()
                        # print(f'current epsilon{i_episode}: {args.epsilon}')
                        random.shuffle(shortest_path_pool.buffer)

                        for action in shortest_path_pool.buffer:
                            print(action, end='->')
                            # env.render()
                            next_state, reward, done1, done2, info = env.step(action)
                            done = done1 or done2
                            episode_buffer.append((state, action, reward, next_state, done, info['new_rms']))
                            PriorBuffer1.add(state, action, reward, next_state, done)
                            buffer_sample_training(PriorBuffer1, args.minimal_size, agent)
                            state = next_state
                            episode_return += reward
                            initial_state = False
                else:
                    state, reset_info = env.reset()
                    initial_state = True
                done = False

                while not done and ((short_path_min_rms != 0 and not initial_state) or initial_state):
                    if initial_state:
                        action = agent.take_action(state)
                    else:
                        action = agent.take_action2()
                    action_record.append(action)
                    # env.render()
                    next_state, reward, done1, done2, info = env.step(action)
                    done = done1 or done2
                    if info['action_valid']:
                        episode_buffer.append((state, action, reward, next_state, done, info['new_rms']))

                    PriorBuffer1.add(state, action, reward, next_state, done)
                    buffer_sample_training(PriorBuffer1, args.minimal_size, agent)
                    state = next_state
                    episode_return += reward

                if short_path_min_rms == 0 and not initial_state:
                    for exp in episode_buffer:
                        state, action, reward, next_state, done = exp
                        episode_return += reward
                        PriorBuffer1.add(state, action, reward, next_state, done)
                        buffer_sample_training(PriorBuffer1, args.minimal_size, agent)

                if short_path_min_rms > info['min_rms']:
                    shortest_path_pool.clear()
                    rewards = []

                    for a in range(info['change_t']):
                        state, action, reward, next_state, done, new_rms = episode_buffer[a]
                        if a == 0:
                            rms_dif = reset_info['initial_rms'] - new_rms
                            old_rms = new_rms
                        else:
                            rms_dif = old_rms - new_rms
                            old_rms = new_rms
                        rewards.append((action, rms_dif))

                    print('Before optimization:', rewards)
                    rewards.sort(key=lambda x: x[1], reverse=True)

                    for action, _ in rewards[:]:
                        shortest_path_pool.add(action)
                    # print('Priority ranking:', shortest_path_pool.buffer)

                    shortest_path_pool.rms_record(info['min_rms'])
                    short_path_min_rms = info['min_rms']

                    shortest_path_simple = optimize_actions(shortest_path_pool.buffer)
                    shortest_path_pool.clear()
                    for action_sim in shortest_path_simple:
                        shortest_path_pool.add(action_sim)
                    print('After optimization:', shortest_path_pool.buffer)

                agent.loss_record()
                agent.q_value_record()
                return_list.append(episode_return)

                if (i_episode + 1) % 5 == 0:
                    pbar.set_postfix({
                        'episode': f'{int(args.num_episodes / 10 * i + i_episode + 1)}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)

    torch.save(agent.q_net.state_dict(), rf'{args.task_batch_path[0]}\results\model\your_model.pth')

    plt.figure()
    plt.plot(range(len(agent.epoch_losses)), agent.epoch_losses)
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for Epoch {i}')
    plt.savefig(rf'{args.task_batch_path[0]}\results\epoch_loss_plot\loss_plot{i}.png')
    plt.close()

    plt.figure()
    plt.plot(range(len(agent.epoch_losses)), rl_utils.moving_average(agent.epoch_losses, 19))
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for Epoch {i}')
    plt.savefig(rf'{args.task_batch_path[0]}\results\epoch_loss_moving_average_plot\moving_average_loss_plot{i}.png')
    plt.close()

    plt.figure()
    plt.plot(range(len(return_list)), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on gym_examples/GridWorld-v0')
    plt.savefig(rf'{args.task_batch_path[0]}\results\epoch_reward_plot\return_plot{i}.png')
    plt.close()

    plt.figure()
    plt.plot(range(len(return_list)), rl_utils.moving_average(return_list, 19))
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on gym_examples/GridWorld-v0')
    plt.savefig(rf'{args.task_batch_path[0]}\results\epoch_reward_average_plot\moving_average_plot{i}.png')
    plt.close()

    with open(rf'{args.task_batch_path[0]}\results\losses.txt', 'w') as f:
        for label, loss in enumerate(agent.epoch_losses):
            f.write(f"{label},{loss}\n")

    with open(rf'{args.task_batch_path[0]}\results\return_list.txt', 'w') as f:
        for label, return_value in enumerate(return_list):
            f.write(f"{label},{return_value}\n")

    state, reset_info = env.reset(best_result=reset_info['best_result'])
    print('Program ended:', 'The optimal parameter values are:', reset_info['best_result'], 'The smallest rmse is:',
          reset_info['min_rms'],
          'The model has been set to optimal.')
    return reset_info['best_result']
