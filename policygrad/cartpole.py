from agent import Agent
import numpy as np
import gym
 
if __name__=='__main__':
    env=gym.make('CartPole-v1')
    agent=Agent(lr=0.001,input_dims=[4],gamma=0.99,n_actions=2)
    score_history=[]
    score=0
    n_episodes=1000
    for i in range(n_episodes):
        print('episode',i,'score %.3f' % score)
        done=False
        score=0
        observation=env.reset()
        
        while not done:
            #env.render()
            action=agent.choose_action(observation)
            observation_new,reward,done,info=env.step(action)
            agent.store_rewards(reward)
            observation=observation_new
            score+=reward
        score_history.append(score)
        agent.learn()

