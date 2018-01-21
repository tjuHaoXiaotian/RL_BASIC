import gym
import tensorflow as tf
from critic import Critic
from actor import Actor
from experience_replay import Experience_replay
import numpy as np


class Train:
    def __init__(self, action_dim, state_dim):
        self.sess = tf.Session()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = 32
        self.episodes = 0
        self.iterations = 0

        self.critic = Critic(self.sess, self.action_dim, self.state_dim)
        self.actor = Actor(self.sess, self.action_dim, self.state_dim)
        self.ER = Experience_replay(500000, self.action_dim, self.state_dim)
        self.sess.run(tf.global_variables_initializer())
        self.critic.operation_update_TDnet_compeletely()
        self.actor.operation_update_TDnet_compeletely()
        self.env = gym.make('Pendulum-v0')

    def add_noise_and_reshape(self, action, var):
        return np.clip(np.random.normal(action, var), -1., 1.).reshape(self.action_dim)

    def operation_add_memory_by_episode(self,episodes,max_iters,var):
        for i in range(episodes):
            observation = self.env.reset()
            for j in range(max_iters):
                # env.render()
                action = self.actor.operation_get_action_to_environment(np.reshape(observation, (1, self.state_dim)))
                action_noise = self.add_noise_and_reshape(action, var)
                observation_next, reward, done, _ = self.env.step(action_noise * 2)  # a [-2,2]
                self.ER.experience_in((observation, action_noise, (reward + 5) / 100., observation_next, done))
                observation = observation_next
                if done:
                    #print 'done'
                    break

    def operation_train_actor(self,s):
        action = self.actor.operation_get_action_to_environment(s)
        summary_g, gradient = self.critic.operation_get_gradient(action, s)
        self.actor.operation_actor_learn(gradient, s)
        return summary_g

    def operation_train_critic(self,s,a,r,ss,t):
        action_next = self.actor.operation_get_action_to_TDtarget(ss)
        td_target = self.critic.operation_get_TDtarget(action_next, ss, r, t)
        summary_c, summary_Q = self.critic.operation_critic_learn(td_target, a, s)
        return summary_c, summary_Q

    def operation_write_summary(self,summary_g,summary_c,summary_Q,iters):
        self.critic.writer.add_summary(summary_g, iters)
        self.critic.writer.add_summary(summary_c, iters)
        self.critic.writer.add_summary(summary_Q, iters)

    def update_T_net(self):
        self.actor.operation_update_TDnet()
        self.critic.operation_update_TDnet()


train = Train(1,3)
train.operation_add_memory_by_episode(500,200,0.4)#add memory
for train.episodes in range(10000000):
    train.operation_add_memory_by_episode(1, 200, 0.4)#perceive
    for i in range(25):
        s,a,r,ss,t = train.ER.experience_out(train.batch_size)#sample
        actor_s = train.ER.experience_out_partly(train.batch_size,100000)#sample

        summary_c,summary_Q = train.operation_train_critic(s,a,r,ss,t)#critic learn

        summary_g = train.operation_train_actor(actor_s)#actor learn
        train.update_T_net()#update t net
        if i % 9 == 0:#write summary   2 times
            train.operation_write_summary(summary_g,summary_c,summary_Q,train.iterations)
        train.iterations += 1