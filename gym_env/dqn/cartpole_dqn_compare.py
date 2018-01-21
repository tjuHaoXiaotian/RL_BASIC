import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from cartpole_dqn_v1 import DQN as DQN1
from cartpole_dqn_v2 import DQN as DQN2


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation
STEP = 200 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
TARGET_Q_UPDATE_FREQUENCY = 5

RENDER_TEST = False

def main():
  # initialize OpenAI Gym env and dqn agent
  env = gym.make(ENV_NAME)
  tests1,tests2 = {}, {}

  plt.figure(figsize=(6, 5))
  plt.subplot(111)
  plt.ion()  # interactive mode on

  with tf.Session() as sess:
    agent1 = DQN1(env,sess)
    agent2 = DQN2(env,sess)
    every_step = 0
    for episode in range(EPISODE):
      # Train agent 2
      # initialize task
      state = env.reset()
      # while True:
      for step in range(STEP):
        every_step += 1
        action = agent2.egreedy_action(state) # e-greedy action for train
        next_state,reward,done,_ = env.step(action)
        # Define reward for agent
        reward_agent = -1 if done else 0.1
        agent2.perceive(state,action,reward,next_state,done)
        state = next_state
        # 同步两个网络
        if every_step == TARGET_Q_UPDATE_FREQUENCY:
          agent2.Q_target.copy(agent2.Q, sess)
          every_step = 0
        if done:
          break

      # Train agent 1
      # initialize task
      state = env.reset()
      # while True:
      for step in range(STEP):
        action = agent1.egreedy_action(state)  # e-greedy action for train
        next_state, reward, done, _ = env.step(action)
        # Define reward for agent
        reward_agent = -1 if done else 0.1
        agent1.perceive(state, action, reward, next_state, done)
        state = next_state
        # 同步两个网络
        if done:
          break

      # update the learning parameter every episode
      agent1.update_learning_parameters()
      agent2.update_learning_parameters()

      # Test every 100 episodes
      if episode % 100 == 0:

        total_reward1 = 0
        for i in range(TEST):
          state = env.reset()
          for j in range(STEP):
            if RENDER_TEST:
              env.render()
            action = agent1.action(state) # direct action for test
            state,reward,done,_ = env.step(action)
            total_reward1 += reward
            if done:
              break
        ave_reward1 = total_reward1/TEST
        print('episode: ',episode,'Evaluation Average Reward(agent 1): ',ave_reward1)
        tests1[episode] = ave_reward1
        # if ave_reward1 >= 200:  # 最大应该是 STEP × 1 = 300
        #   break

        total_reward2 = 0
        for i in range(TEST):
          state = env.reset()
          for j in range(STEP):
            if RENDER_TEST:
              env.render()
            action = agent2.action(state)  # direct action for test
            state, reward, done, _ = env.step(action)
            total_reward2 += reward
            if done:
              break
        ave_reward2 = total_reward2 / TEST
        print('episode: ', episode, 'Evaluation Average Reward(agent 2): ', ave_reward2)
        tests2[episode] = ave_reward2
      # draw the learning process plot

      x1 = [key for key in tests1]
      x1 = sorted(x1)
      y1 = [tests1[key] for key in x1]

      x2 = [key for key in tests2]
      x2 = sorted(x2)
      y2 = [tests2[key] for key in x2]

      plt.clf()
      plt.xlabel('Steps')
      plt.ylabel('Avg rewards of one episode')
      plt.xlim(0, EPISODE)
      # plt.ylim(0, max(y))
      plt.ylim(0, STEP)
      plt.plot(x1, y1, label='average rewards of 10 test episode(DQN1)', color='red', linewidth=1)
      plt.plot(x2, y2, label='average rewards of 10 test episode(DQN2)', color='green', linewidth=1)
      plt.legend(loc='lower right')
      # clf()  # 清图。
      # cla()  # 清坐标轴。
      # close()  # 关窗口
      plt.pause(0.001)
      # plt.show()



if __name__ == '__main__':
  main()
  tf.squeeze()