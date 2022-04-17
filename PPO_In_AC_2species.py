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
tf.compat.v1.disable_eager_execution()

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
        self.state_input = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.state_dim], 'state')
        # Init session
        self.old_r = 0
        self.Critic_LR = critic_lr
        self.Actor_LR = actor_lr
        self.Target_UPDATE_TIMES = target_episodes
        self.ACTOR_UPDATE_TIMES = actor_episodes
        self.CRITIC_UPDATE_TIMES = critic_episodes
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.Create_Critic()
        self.Create_Actor_with_two_network()
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        self.model_name = name
        # 判断结果
        if not os.path.exists('PPO_model/'+self.model_name):
            os.makedirs('PPO_model/'+self.model_name)
            print(self.model_name + ' 创建成功')
        else:
            print(self.model_name + ' 目录已存在')

    # the critic network give the value of state
    def Create_Critic(self):
        # first, create the parameters of networks
        W1 = self.weight_variable([self.state_dim, self.fc1_size])
        b1 = self.bias_variable([self.fc1_size])
        W2 = self.weight_variable([self.fc1_size, self.fc2_size])
        b2 = self.bias_variable([self.fc2_size])
        W3 = self.weight_variable([self.fc2_size, self.action_dim])
        b3 = self.bias_variable([self.action_dim])
        h_layer_one = tf.compat.v1.nn.relu(tf.compat.v1.matmul(self.state_input, W1) + b1)
        h_layer_two = tf.compat.v1.nn.relu(tf.compat.v1.matmul(h_layer_one, W2) + b2)
        self.v = tf.compat.v1.matmul(h_layer_two, W3) + b3
        # third, give the update method of critic network
        # the input of discounted reward
        self.tfdc_r = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.advantage))
        self.ctrain_op = tf.compat.v1.train.AdamOptimizer(self.Critic_LR).minimize(self.closs)
        return

    # the actor network that give the action
    def Create_Actor_with_two_network(self):
        # create the actor that give the action distribution
        pi, pi_params = self.build_actor_net('pi', trainable=True)
        oldpi, oldpi_params = self.build_actor_net('oldpi', trainable=False)
        # sample the action from the distribution
        with tf.compat.v1.variable_scope('sample_action'):
            # self.sample_from_pi = tf.compat.v1.squeeze(pi.sample(1), axis=0)
            self.sample_from_oldpi = tf.compat.v1.squeeze(oldpi.sample(1), axis=0)
        # update the oldpi by coping the parameters from pi
        with tf.compat.v1.variable_scope('update_oldpi'):
            self.update_oldpi_from_pi = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
        # the actions in memory
        self.tfa = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.action_dim], 'action')
        # the advantage value
        self.tfadv = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, self.action_dim], 'advantage')
        with tf.compat.v1.variable_scope('loss'):
            with tf.compat.v1.variable_scope('surrogate'):
                # the ration between the pi and oldpi, this is importance sampling part
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                surr = ratio * self.tfadv
            self.aloss = -tf.compat.v1.reduce_mean(tf.compat.v1.minimum(
                surr,
                tf.compat.v1.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv))
        # define the method of training actor
        with tf.compat.v1.variable_scope('atrain'):
            self.atrain_op = tf.compat.v1.train.AdamOptimizer(self.Actor_LR).minimize(self.aloss)
        return

    def build_actor_net(self, name, trainable):
        with tf.compat.v1.variable_scope(name):
            l1 = tf.compat.v1.layers.dense(self.state_input, self.fc1_size, tf.compat.v1.nn.relu, trainable=trainable)
            l2 = tf.compat.v1.layers.dense(l1, self.fc2_size, tf.compat.v1.nn.relu, trainable=trainable)
            mu = (tf.compat.v1.layers.dense(l2, self.action_dim, tf.compat.v1.nn.tanh, trainable=trainable) + 1) / 2
            sigma = tf.compat.v1.layers.dense(l2, self.action_dim, tf.compat.v1.nn.softplus, trainable=trainable)
            norm_dist = tf.compat.v1.distributions.Normal(loc=mu, scale=sigma)
        params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # output the action with state, the output is from oldpi
    def Choose_Action(self, s):
        # s = s[np.newaxis, :]
        a = self.sess.run(self.sample_from_oldpi, {self.state_input: s})[0]
        return np.clip(a, 0, 1)

    # reset the memory in every episode
    def resetMemory(self):
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []

    # store the data of every steps
    def Store_Data(self, state, action, reward, next_state, done):
        self.buffer_s.append(state)
        self.buffer_a.append(action)
        self.buffer_r.append(reward)

    # get the state value from critic
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.state_input: s})[0, 0]

    # the function that update the actor and critic
    def update(self, s, a, r):
        adv = self.sess.run(self.advantage, {self.state_input: s, self.tfdc_r: r})
        # update critic
        [self.sess.run(self.ctrain_op, {self.state_input: s, self.tfdc_r: r})
         for _ in range(self.CRITIC_UPDATE_TIMES)]
        # update actor
        [self.sess.run(self.atrain_op, {self.state_input: s, self.tfa: a, self.tfadv: adv})
         for _ in range(self.ACTOR_UPDATE_TIMES)]

    # the train function that update the network
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
        self.sess.run(self.update_oldpi_from_pi)
        if r > self.old_r:
            self.saver.save(self.sess, save_path='PPO_model/'+self.model_name+'/'+self.model_name)
            self.old_r = r

    # the function that give the weight initial value
    def weight_variable(self, shape):
        initial = tf.compat.v1.truncated_normal(shape)
        return tf.compat.v1.Variable(initial)

    # the function that give the bias initial value
    def bias_variable(self, shape):
        initial = tf.compat.v1.constant(0.01, shape=shape)
        return tf.compat.v1.Variable(initial)

    def load_model(self):
        self.saver.restore(self.sess, 'PPO_model/'+self.model_name+'/'+self.model_name)


class Jet(object):
    def __init__(self):
        self.target_density = np.array([[0]], dtype='float32')
        self.output_density = np.array([[0]*2], dtype='float32')
        self.distance = self.target_density - self.output_density
        self.action1 = np.array([[0]*2], dtype='float32')
        self.action2 = np.array([[0]*2], dtype='float32')
        self.state = np.concatenate((self.distance, self.action1, self.action2), axis=1)
        self.SysModel_NO = tf.keras.models.load_model('DNN/DNN_Model_NO_2')
        self.SysModel_O = tf.keras.models.load_model('DNN/DNN_Model_O_2')
        action_low = np.array([0]*2)
        action_high = np.array([1]*2)
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)
        state_low = np.array([0]*2*(1+2))
        state_high = np.array([1]*2*(1+2))
        self.observation_space = Box(low=state_low, high=state_high, dtype=np.float32)

    def reset(self):
        # restart from 0
        self.output_density = np.array([[0]*2], dtype='float32')
        self.distance = self.target_density - self.output_density
        self.action1 = np.array([[0]*2], dtype='float32')
        self.action2 = np.array([[0]*2], dtype='float32')
        self.state = np.concatenate((self.distance, self.action1, self.action2), axis=1)
        return self.state

    def step(self, action, dis=0):
        terminal = False
        # choose flow and distance
        act = np.concatenate((action, [dis], [0, 0, 0.8])).reshape(1, 6)
        density_no = np.squeeze(self.SysModel_O.predict(act))
        # density_o = np.squeeze(self.SysModel_O.predict(act))
        self.output_density = np.roll(self.output_density, 1)
        self.output_density[0][0] = density_no
        self.action1 = np.roll(self.action1, 1)
        self.action1[0][0] = np.squeeze(action[0])
        self.action2 = np.roll(self.action2, 1)
        self.action2[0][0] = np.squeeze(action[1])
        self.distance = self.target_density - self.output_density
        self.state = np.concatenate((self.distance, self.action1, self.action2), axis=1)
        dis = abs(density_no - self.target_density).sum()
        reward = -dis
        # reward = reward[0, 0]
        if dis < 0.01:
            reward = 20
            terminal = True
        info = action
        return self.state, reward, terminal, info


# the total episodes of training
NUM = 60
OFFSET = 2
seq = [0.5 + 1 / NUM * i for i in range(NUM)] + \
           [0.5 + 1 + 0.2 * np.sin(0.5 * i) for i in range(NUM)] + \
           [2] * NUM + \
           [2.3] * NUM + \
           [0.8] * NUM + \
           [2.6] * NUM  # +\
test_seq = [x/3.5 for x in seq]
ROLL_STEPS = 100
STEPS = 100
EPISODES = 1000
train = 1
# train = True
all_ep_r = []
GAMMA = 0.9
# the update times of actor and critic
Target_UPDATE_TIMES = 20
ACTOR_UPDATE_TIMES = 10
CRITIC_UPDATE_TIMES = 20
FC1_n = 256
FC2_n = 128
Critic_LR = 2e-4
Actor_LR = 1e-4
MODEL_NAME = 'model-2factors-O'
test = np.array([1.5])


def main():
    env = Jet()
    agent = PPO(env.observation_space, env.action_space, Critic_LR, Actor_LR, Target_UPDATE_TIMES,
                ACTOR_UPDATE_TIMES, CRITIC_UPDATE_TIMES, FC1_n, FC2_n, MODEL_NAME)
    target = np.random.uniform(0.15, 0.85, (ROLL_STEPS, 1))
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
            agent.UpdateActorParameters(ep_r)
            all_ep_r.append(ep_r)
            print(
                'Ep: %i' % episode,
                "|Ep_r: %i" % ep_r,
                "action: ", action)
            if episode % 10 == 0 or episode == EPISODES-1:
                outfile = pd.DataFrame(all_ep_r)
                outfile.to_csv('PPO_model/'+MODEL_NAME+'/'+MODEL_NAME + '-epoch.csv', sep=',', header=False, index=False)
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
        plt.plot(np.array(out_v))
        plt.xlabel("Step")
        plt.ylabel("Out_v")
        Set_seq = np.array(test_seq)
        act_list = act_list.reshape(-1, 2)
        outfile = pd.DataFrame(np.concatenate((Set_seq.reshape(-1, 1), out_v.reshape(-1, 1), act_list), axis=1))
        outfile.to_csv('PPO_model/' + MODEL_NAME + '/' + MODEL_NAME + '-track.csv', sep=',')
        plt.show()
    elif train == 3:
        agent.load_model()
        env.target_density = 0.6
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
        act_list = act_list.reshape(-1, 2)
        outfile = pd.DataFrame(np.concatenate((out_v.reshape(-1, 1), act_list), axis=1))
        outfile.to_csv('PPO_model/' + MODEL_NAME + '/' + MODEL_NAME + '-disturb.csv', sep=',', header=False, index=False)
        plt.show()
    return


if __name__ == '__main__':
    main()
