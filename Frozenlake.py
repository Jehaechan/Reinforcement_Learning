import numpy as np

def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    temp = env.MDP
    while True:
        while True:
            delta = 0
            for i in range(env.nS):
                if env.map[i // 4][i % 4] in b'GH':
                    continue
                v = V.copy()
                sum_temp = 0
                for j in range(env.nA):
                    action_temp = temp[i][j]
                    for k in range(len(action_temp)):
                        sum_temp += policy[i][j] * action_temp[k][0] * (action_temp[k][2] + gamma * V[action_temp[k][1]])
                V[i] = sum_temp
                dif = abs(v[i]-V[i])
                if dif > delta:
                    delta = dif
            if delta < theta:
                break

        policy_stable = True
        for i in range(env.nS):
            old_action = policy[i].copy()
            argmax = 0
            temp_j = 0
            for j in range(env.nA):
                action_temp = temp[i][j]
                temp_sum = 0
                for k in range(len(action_temp)):
                    temp_sum += action_temp[k][0] * (action_temp[k][2] + gamma * V[action_temp[k][1]])
                if temp_sum > argmax:
                    argmax = temp_sum
                    temp_j = j
            policy[i] = np.zeros(env.nA)
            policy[i][temp_j] = 1
            if not (policy[i] == old_action).all():
                policy_stable = False

        if policy_stable:
            break

    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    temp = env.MDP

    while True:
        delta = 0
        for i in range(env.nS):
            if env.map[i // 4][i % 4] in b'GH':
                continue
            v = V.copy()

            argmax = 0
            for j in range(env.nA):
                action_temp = temp[i][j]
                temp_sum = 0
                for k in range(len(action_temp)):
                    temp_sum += action_temp[k][0] * (action_temp[k][2] + gamma * V[action_temp[k][1]])
                if temp_sum > argmax:
                    argmax = temp_sum
            V[i] = argmax
            dif = abs(v[i] - V[i])
            if dif > delta:
                delta = dif
        if delta < theta:
            break

    for i in range(env.nS):
        argmax = 0
        temp_j = 0
        for j in range(env.nA):
            action_temp = temp[i][j]
            temp_sum = 0
            for k in range(len(action_temp)):
                temp_sum += action_temp[k][0] * (action_temp[k][2] + gamma * V[action_temp[k][1]])
            if temp_sum > argmax:
                argmax = temp_sum
                temp_j = j
        policy[i] = np.zeros(env.nA)
        policy[i][temp_j] = 1
    return policy, V
