import gym
from gym.spaces import Box
import tensorflow as tf
import pandas as pd
# tf.compat.v1.disable_v2_behavior()
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow_probability.python.distributions import Normal


# the method of updating
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]  # choose the method 'clip' for optimization


# the output of PPO is continuous
class PPO:
    def __init__(self, observation_space, action_space, critic_lr, actor_lr, target_episodes,
                 actor_episodes, critic_episodes, fc1_size, fc2_size, name):
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.old_r = 0
        self.Target_UPDATE_TIMES = target_episodes
        self.ACTOR_UPDATE_TIMES = actor_episodes
        self.CRITIC_UPDATE_TIMES = critic_episodes
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []

        self.Critic = self.Create_Critic()
        self.actor = self.build_actor_net('pi', trainable=True)
        self.old_actor = self.build_actor_net('oldpi', trainable=False)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=actor_lr)  # 优化器
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=critic_lr)  # 优化器
        self.model_name = name
        # 判断结果
        if not os.path.exists('PPO_model/' + self.model_name):
            os.makedirs('PPO_model/' + self.model_name)
            print(self.model_name + ' 创建成功')
        else:
            print(self.model_name + ' 目录已存在')

    # the critic network give the value of state
    def Create_Critic(self):
        # first, create the parameters of networks
        model = tf.keras.Sequential([
            # [b, 8] => [b, 64]
            tf.keras.layers.Dense(self.fc1_size, activation="relu"),
            # [b, 64] => [b, 64]
            tf.keras.layers.Dense(self.fc2_size, activation="relu"),
            # [b, 64] => [b, 1]
            tf.keras.layers.Dense(1)
        ])
        return model

    def build_actor_net(self, name, trainable):
        # 行动
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        act_layer1 = tf.keras.layers.Dense(self.fc1_size, activation="relu")(state_input)
        act_layer2 = tf.keras.layers.Dense(self.fc2_size, activation="relu")(act_layer1)
        mu_out = (tf.keras.layers.Dense(self.action_dim, activation="tanh")(act_layer2) + 1) / 2
        sigma_out = tf.keras.layers.Dense(self.action_dim, activation="softplus")(act_layer2)
        model = tf.keras.models.Model(inputs=state_input, outputs=[mu_out, sigma_out])
        return model

    # output the action with state, the output is from oldpi
    def Choose_Action(self, s):
        action = self.get_a(s)
        return np.clip(action.numpy()[0], 0, 1)

    def get_a(self, s):
        [act_mu, act_sigma] = self.old_actor(s)
        dist = Normal(loc=act_mu, scale=act_sigma)
        action = dist.sample()
        return action

    # get the state value from critic
    def get_v(self, s):
        return self.Critic(s)[0, 0]

    def get_prob(self, s, a):
        [act_mu, act_sigma] = self.actor(s)
        dist = Normal(loc=act_mu, scale=act_sigma)
        # 计算概率密度, log(概率)
        action_probs = dist.prob(a)

        [act_mu, act_sigma] = self.old_actor(s)
        dist = Normal(loc=act_mu, scale=act_sigma)
        # 计算概率密度, log(概率)
        old_action_probs = dist.prob(a)
        return action_probs, old_action_probs

    # the function that update the actor and critic
    @tf.function
    def update(self, s, a, r):
        for _ in range(self.CRITIC_UPDATE_TIMES):
            with tf.GradientTape() as tape:
                adv = r - self.get_v(s)
                closs = tf.reduce_mean(tf.square(adv))
                grads = tape.gradient(closs, self.Critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.Critic.trainable_variables))
        for _ in range(self.ACTOR_UPDATE_TIMES):
            with tf.GradientTape() as tape:
                action_probs, old_action_probs = self.get_prob(s, a)
                ratio = action_probs / (old_action_probs + 1e-5)
                adv = r - self.get_v(s)
                surr = ratio * adv
                aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv))
                grads = tape.gradient(aloss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    # the train function that update the network
    # @tf.function
    def Train(self, next_state):
        # caculate the discount reward
        for i in range(self.Target_UPDATE_TIMES):
            v_s_ = self.get_v(next_state)
            discounted_r = []
            for r in self.buffer_r[::-1]:  # 将reward倒序求解前面的v_s_
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            # discounted_r = discounted_r/(np.std(discounted_r) + 1e-5)
            bs, ba, br = np.vstack(self.buffer_s), np.vstack(self.buffer_a), np.array(discounted_r)[:, np.newaxis]
            self.update(bs, ba, br)

    # ths dunction the copy the pi's parameters to oldpi
    def UpdateActorParameters(self, r):
        self.old_actor = self.actor
        # if r > self.old_r:
        #     self.saver.save(self.sess, save_path='PPO_model/' + self.model_name + '/' + self.model_name)
        #     self.old_r = r

    # reset the memory in every episode
    def resetMemory(self):
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []

    # store the data of every steps
    def Store_Data(self, state, action, reward, next_state, done):
        self.buffer_s.append(state)
        self.buffer_a.append(action)
        self.buffer_r.append(reward)

    def load_model(self):
        self.saver.restore(self.sess, 'PPO_model/' + self.model_name + '/' + self.model_name)


class Jet(object):
    def __init__(self):
        self.target_density = np.array([[0]], dtype='float32')
        self.output_density = np.array([[0] * hist_num], dtype='float32')
        self.distance = self.target_density - self.output_density
        self.action = np.array([[0] * hist_num * act_num], dtype='float32')
        self.state = np.concatenate((self.distance, self.action), axis=1)
        self.SysModel_NO = tf.keras.models.load_model('DNN_model/DNN_Model_NO_2')
        self.SysModel_O = tf.keras.models.load_model('DNN_model/DNN_Model_O_2')
        action_low = np.array([0] * act_num)
        action_high = np.array([1] * act_num)
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)
        state_low = np.array([0] * hist_num * (1 + act_num))
        state_high = np.array([1] * hist_num * (1 + act_num))
        self.observation_space = Box(low=state_low, high=state_high, dtype=np.float32)

    def reset(self):
        # restart from 0
        self.output_density = np.array([[0] * hist_num], dtype='float32')
        self.distance = self.target_density - self.output_density
        self.action = np.array([[0] * hist_num * act_num], dtype='float32')
        self.state = np.concatenate((self.distance, self.action), axis=1)
        return self.state

    @tf.function
    def step(self, action, vol=1, flow=0.5, inter=0.33, p_o2=0, p_h2o=0, fre=1):
        terminal = False
        # choose flow and distance
        # p_o2 = action[2]
        # inter = action[3]
        act = np.array([vol, flow, inter, p_o2, p_h2o, fre]).reshape(1, 6)
        act[0, 0:act_num] = action
        # density_no = np.squeeze(self.SysModel_NO.predict(act))
        density_o = np.squeeze(self.SysModel_O.predict(act))
        self.output_density = np.roll(self.output_density, 1)
        self.output_density[0][0] = density_o
        for i in range(hist_num - 1):
            self.action[0, hist_num - i - 1:hist_num * act_num:hist_num] = self.action[0,
                                                                           hist_num - i - 2:hist_num * act_num:hist_num]
        self.action[0, 0:hist_num * act_num:hist_num] = action
        self.distance = self.target_density - self.output_density
        self.state = np.concatenate((self.distance, self.action), axis=1)
        dis = abs(density_o - self.target_density).sum()
        reward = -dis
        # reward = reward[0, 0]
        if dis / self.target_density < 0.04:
            reward = 20
        elif dis / self.target_density < 0.07:
            reward = 10
        elif dis / self.target_density < 0.1:
            reward = 5
        info = action
        return self.state, reward, terminal, info


# the total episodes of training
act_num = 3
hist_num = 2
NUM = 60
OFFSET = 2
test_seq = [0.15 + 0.3 / NUM * i for i in range(NUM)] + \
           [0.15 + 0.3 + 0.1 * np.sin(0.5 * i) for i in range(NUM)] + \
           [0.55] * NUM + \
           [0.65] * NUM + \
           [0.2] * NUM + \
           [0.7] * NUM  # +\
# test_seq = [x/3.5 for x in seq]
ROLL_STEPS = 50
STEPS = 100
EPISODES = 1000
train = 1
# train = True
all_ep_r = []
GAMMA = 0.8
# the update times of actor and critic
Target_UPDATE_TIMES = 10
ACTOR_UPDATE_TIMES = 5
CRITIC_UPDATE_TIMES = 10
FC1_n = 128
FC2_n = 128
Critic_LR = 2e-4
Actor_LR = 1e-4
MODEL_NAME = 'model-2actions-O-tf2'
test = np.array([1.5])


def main():
    env = Jet()
    agent = PPO(env.observation_space, env.action_space, Critic_LR, Actor_LR, Target_UPDATE_TIMES,
                ACTOR_UPDATE_TIMES, CRITIC_UPDATE_TIMES, FC1_n, FC2_n, MODEL_NAME)
    target = np.random.uniform(0, 0.8, (ROLL_STEPS, 1))
    if train == 1:
        for episode in range(EPISODES):
            agent.resetMemory()
            ep_r = 0
            for roll in range(ROLL_STEPS):
                env.target_density = np.array([target[roll, 0]])
                state = env.reset()
                for step in range(STEPS):
                    action = agent.Choose_Action(state)
                    next_state, reward, done, info = env.step(action)
                    agent.Store_Data(state, action, reward, next_state, done)
                    state = next_state
                    ep_r += reward
            agent.Train(next_state)
            ep_r = ep_r / ROLL_STEPS
            agent.UpdateActorParameters(ep_r)
            all_ep_r.append(ep_r)
            print(
                'Ep: %i' % episode,
                "|Ep_r: %i" % ep_r,
                "action: ", action)
            if episode % 10 == 0 or episode == EPISODES - 1:
                outfile = pd.DataFrame(all_ep_r)
                outfile.to_csv('PPO_model/' + MODEL_NAME + '/' + MODEL_NAME + '-epoch.csv', sep=',', header=False,
                               index=False)
            # if ep_r > 40000:
            #     break
        plt.plot(all_ep_r)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
    elif train == 2:
        agent.load_model()
        env.target_density = test_seq[0]
        state = env.reset()
        out_v = np.array([])
        act_list = np.array([])
        for T in test_seq:
            # output the action
            env.target_density = T
            action = agent.Choose_Action(state)
            # process the action and get the info
            next_state, reward, done, info = env.step(action)
            # record the reward
            out_v = np.append(out_v, env.target_density - state[0][0])
            act_list = np.append(act_list, action)
            state = next_state
        fig, ax = plt.subplots()
        x = np.linspace(1, 360, 360)
        ax.plot(x, np.array(out_v), label='out')
        ax.plot(x, np.array(test_seq), label='set')
        ax.set_xlabel("Step")
        ax.set_ylabel("density")
        ax.legend()
        set_seq = np.array(test_seq)
        act_list = act_list.reshape(-1, act_num)
        outfile = pd.DataFrame(np.concatenate((set_seq.reshape(-1, 1), out_v.reshape(-1, 1), act_list), axis=1))
        outfile.to_csv('PPO_model/' + MODEL_NAME + '/' + MODEL_NAME + '-track.csv', sep=',')
        plt.show()
    elif train == 3:
        agent.load_model()
        env.target_density = 0.25
        state = env.reset()
        out_v = np.array([])
        act_list = np.array([])
        for T in range(200):
            # output the action
            action = agent.Choose_Action(state)
            # process the action and get the info
            next_state, reward, done, info = env.step(action, dis=0.33 * (T // 50))
            # record the reward
            out_v = np.append(out_v, env.target_density - state[0][0])
            state = next_state
            act_list = np.append(act_list, action)
        plt.plot(np.array(out_v))
        plt.xlabel("Step")
        plt.ylabel("Out_v")
        act_list = act_list.reshape(-1, act_num)
        outfile = pd.DataFrame(np.concatenate((out_v.reshape(-1, 1), act_list), axis=1))
        outfile.to_csv('PPO_model/' + MODEL_NAME + '/' + MODEL_NAME + '-disturb.csv', sep=',', header=False,
                       index=False)
        plt.show()
    return


if __name__ == '__main__':
    main()
