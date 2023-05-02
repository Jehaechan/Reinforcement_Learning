import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp
import time
import numpy as np
import math

ENV = gym.make("InvertedPendulumSwingupBulletEnv-v0")
OBS_DIM = ENV.observation_space.shape[0]    # 5
ACT_DIM = ENV.action_space.shape[0]         # 1
ACT_LIMIT = ENV.action_space.high[0]        # 1.0
ENV.close()

MAX_EP = 3000
GAMMA = 0.95

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.a1 = nn.Linear(OBS_DIM, 200)
        self.mu = nn.Linear(200, ACT_DIM)
        self.sigma = nn.Linear(200, ACT_DIM)
        self.c1 = nn.Linear(OBS_DIM, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = Normal(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss

    def act(self, x):
        mu, std, trash = self.forward(x)
        return mu, std


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)


                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def push_and_pull(opt, agent, gnet, bs, ba, br, gamma, n_steps):
    buffer_v_target = []
    reversed_br = list(reversed(br))
    reversed_bs = list(reversed(bs))

    for i in range(UPDATE_GLOBAL_ITER):
        v_s_ = agent.forward(v_wrap(reversed_bs[i][None, :]))[-1].data.numpy()[0, 0]
        for j in range(n_steps):
            v_s_ = reversed_br[1 + i + j] + gamma * v_s_
        buffer_v_target.append(v_s_)

    buffer_v_target.reverse()

    loss = agent.loss_func(
        v_wrap(np.vstack(bs[n_steps:-1])),
        v_wrap(np.vstack(ba[n_steps:-1])),
        v_wrap(np.array(buffer_v_target)[:, None]))

    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(agent.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    agent.load_state_dict(gnet.state_dict())

def Worker(global_actor, n_steps):
    env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
    gnet = global_actor
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.9, 0.99))
    agent = ActorCritic()
    agent.train()

    global UPDATE_GLOBAL_ITER
    if n_steps == 1:
        UPDATE_GLOBAL_ITER = 1
    else:
        UPDATE_GLOBAL_ITER = 10

    buffer_size = UPDATE_GLOBAL_ITER + n_steps + 1


    total_step = 1
    for episode in range(MAX_EP):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        while True:
            a = agent.choose_action(v_wrap(s[None, :]))
            s_, r, done, _ = env.step(a.clip(-1, 1))

            buffer_a.append(a)
            buffer_s.append(s)
            buffer_r.append(r)

            if len(buffer_a) > buffer_size:
                buffer_a.pop(0)
                buffer_s.pop(0)
                buffer_r.pop(0)


            if (total_step == buffer_size) or (total_step % UPDATE_GLOBAL_ITER) == 0 or done:
                if len(buffer_a) == buffer_size:
                    push_and_pull(opt, agent, gnet, buffer_s, buffer_a, buffer_r, GAMMA, n_steps)

            if done:
                break

            s = s_
            total_step += 1


    env.close()
    print("Training process reached maximum episode.")
