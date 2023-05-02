import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque


def get_demo_traj():
    return np.load("demo_traj_2.npy", allow_pickle=True)

class DQfDNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(DQfDNetwork, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.layer1 = 128
        self.layer2 = 256

        self.fc1 = nn.Linear(self.in_size, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    def __init__(self):
        self.buffer = deque(maxlen=10000)

    def write(self, transition):
        self.buffer.append(transition)

    def sample(self, n, use_per=False):  # s, a, r, n_s, d, n_n_s, sum_r, n_d
        if not use_per:
            mini_batch = random.sample(self.buffer, n)
            s_list, a_list, r_list, n_s_list, d_list, n_n_s_list, sum_r_list, n_d_list = [], [], [], [], [], [], [], []

            for transition in mini_batch:
                s, a, r, n_s, d, n_n_s, sum_r, n_d = transition
                s_list.append(s)
                a_list.append([a])
                r_list.append([r])
                n_s_list.append(n_s)
                d_list.append([1 if not d else 0])
                n_n_s_list.append(n_n_s)
                sum_r_list.append([sum_r])
                n_d_list.append([1 if not n_d else 0])

            ret = torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
                  torch.tensor(r_list), torch.tensor(n_s_list, dtype=torch.float), \
                  torch.tensor(d_list), torch.tensor(n_n_s_list, dtype=torch.float), \
                  torch.tensor(sum_r_list), torch.tensor(n_d_list)

            return ret

    def size(self):
        return len(self.buffer)

def set_seed():
    seed = 38
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class DQfDAgent(object):
    def __init__(self, env, use_per=False, n_episode=250):
        set_seed()

        self.env = env
        self.in_size = env.observation_space.shape[0]
        self.out_size = env.action_space.n
        self.q = DQfDNetwork(self.in_size, self.out_size)
        self.q_target = DQfDNetwork(self.in_size, self.out_size)
        self.q_target.load_state_dict(self.q.state_dict())

        self.n_EPISODES = n_episode
        self.use_per = False

        self.replay_buffer = ReplayMemory()
        self.replay_buffer_demo = ReplayMemory()

        self.eps = 0.01
        self.gamma = 0.98

        self.n_step = 10
        self.n_step_buffer = deque(maxlen=self.n_step)

    def get_action(self, obs):
        out = self.q.forward(obs)
        coin = random.random()
        if coin < self.eps:
            return random.randint(0, 1)
        else:
            return out.argmax().item()

    def pretrain(self):
        demo_list = get_demo_traj()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=0.005, weight_decay=0.00001)

        for demo in demo_list:
            for i in range(len(demo)):
                flag = 1
                reward = 0
                for j in reversed(range(self.n_step)):
                    if i + j > len(demo) - 1:
                        continue
                    if flag:
                        next_state = demo[i + j][3]
                        done = demo[i + j][4]
                        flag = 0
                    reward = reward * self.gamma + demo[i + j][2]
                temp = demo[i][0], demo[i][1], demo[i][2], demo[i][3], demo[i][4], next_state, reward, \
                       done
                # s, a, r, n_s, d, n_n_s, sum_r, n_d
                self.replay_buffer_demo.write(temp)

        for i in range(1000):
            if not self.use_per:
                s, a, r, n_s, d, n_n_s, sum_r, n_d = self.replay_buffer_demo.sample(64, self.use_per)

            s_q = self.q.forward(s)
            q_v = self.q.forward(s).gather(1, a)

            target = r + 0.95 * self.q_target.forward(n_s).max(1)[0].view(-1, 1) * d

            JDQ = F.mse_loss(q_v, target)

            max_q_a = torch.argmax(s_q, -1).view(-1, 1)
            max_q = torch.max(s_q, -1)[0].view(-1, 1)

            JE = (torch.max(s_q, -1)[0].view(-1, 1) + torch.where(max_q_a == max_q, torch.tensor(0.0).float(), torch.tensor(0.8).float()) - q_v).mean()

            JN = (sum_r + (0.95 ** 10) * self.q.forward(n_n_s) * n_d).mean()  # 10 self.n_step?

            loss = JDQ + JE + JN

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 10 == 0:
                self.q_target.load_state_dict(self.q.state_dict())
    ## Do pretrain for 1000 steps

    def agent_train(self):
        for i in range(10):
            if not self.use_per:
                ratio = self.replay_buffer_demo.size() / (self.replay_buffer_demo.size() + self.replay_buffer.size())

                demo_data = int(ratio * 64)

                qs, qa, qr, qnext_state, qd, qn_next_state, qn_r, qnd = self.replay_buffer_demo.sample(demo_data, self.use_per)
                ps, pa, pr, pnext_state, pd, pn_next_state, pn_r, pnd = self.replay_buffer.sample(64 - demo_data, self.use_per)

                pnext_state = torch.squeeze(pnext_state)

            a = torch.cat((qa, pa), 0)
            r = torch.cat((qr, pr), 0)
            next_state = torch.cat((qnext_state, pnext_state), 0)
            d = torch.cat((qd, pd), 0)
            n_next_state = torch.cat((qn_next_state, pn_next_state), 0)
            n_r = torch.cat((qn_r, pn_r), 0)
            nd = torch.cat((qnd, pnd), 0)

            qs_q = self.q.forward(qs)
            ps_q = self.q.forward(ps)
            s_q = torch.cat((qs_q, ps_q), 0)

            q_v = s_q.gather(1, a)
            qq_v = qs_q.gather(1, qa)

            next_s_q_max = self.q_target.forward(next_state).max(1)[0].view(-1, 1)

            target = r + 0.98 * next_s_q_max * d

            JDQ = F.mse_loss(q_v, target)

            max_q_a = torch.argmax(qs_q, -1).view(-1, 1)
            max_q = torch.max(qs_q, -1)[0].view(-1, 1)
            JE_D = (max_q + torch.where(max_q_a == max_q, torch.tensor(0.0).float(), torch.tensor(0.8).float()) - qq_v)
            JE_A = torch.zeros(64 - demo_data, 1)
            JE = torch.cat((JE_D, JE_A), 0)
            JE = JE.mean()

            JN = (n_r + (0.98 ** 10) * self.q.forward(n_next_state) * nd).mean()

            loss = JDQ + JE + JN

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i % 10 == 0):
                self.q_target.load_state_dict(self.q.state_dict())

    def train(self):
        ###### 1. DO NOT MODIFY FOR TESTING ######
        test_mean_episode_reward = deque(maxlen=20)
        test_over_reward = False
        test_min_episode = np.inf
        ret_reward = []
        ###### 1. DO NOT MODIFY FOR TESTING ######

        # Do pretrain
        self.pretrain()
        ## TODO

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=0.00001, weight_decay=0.00001)

        for e in range(self.n_EPISODES):

            ########### 2. DO NOT MODIFY FOR TESTING ###########
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########

            ## TODO
            done = False
            state = self.env.reset()

            while not done:
                ## TODO
                action = self.get_action(torch.tensor(state).float())
                ## TODO
                next_state, reward, done, _ = self.env.step(action)

                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward
                ########### 3. DO NOT MODIFY FOR TESTING  ###########

                ## TODO
                buf_input = state, action, next_state, reward, done
                self.n_step_buffer.append(buf_input)

                if len(self.n_step_buffer) == self.n_step or done:
                    reward = 0
                    is_done = False
                    for i in range(len(self.n_step_buffer) - 1, -1, -1):
                        if self.n_step_buffer[i][4]:
                            reward = 0
                            is_done = True
                        reward = reward * self.gamma + 1

                    if not is_done:
                        temp = self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[0][3], \
                               self.n_step_buffer[0][2], self.n_step_buffer[0][4], self.n_step_buffer[9][
                                   2], reward, False
                    else:
                        for i in range(len(self.n_step_buffer) - 1, -1, -1):
                            if self.n_step_buffer:
                                temp = self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[0][3], \
                                       self.n_step_buffer[0][2], self.n_step_buffer[0][4], self.n_step_buffer[i][
                                           2], reward, True
                                break

                    self.replay_buffer.write(temp)

                state = next_state

                # s, a, r, n_s, d, n_n_s, sum_r, n_d
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward) == 20):
                        test_over_reward = True
                        test_min_episode = e
                    ########### 4. DO NOT MODIFY FOR TESTING  ###########
                    ret_reward.append(test_episode_reward)
                    if self.replay_buffer.size() > 100:
                        self.agent_train()
                ## TODO

            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if test_over_reward:
                print("END train function")
                break
            ########### 5. DO NOT MODIFY FOR TESTING  ###########

            ## TODO

        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)#, ret_reward
        ########### 6. DO NOT MODIFY FOR TESTING  ###########

