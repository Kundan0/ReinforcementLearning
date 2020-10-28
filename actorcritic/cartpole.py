from agent import Agent
#from utils import plotLearning
import gym
import matplotlib.pyplot as plt
 
if __name__=='__main__':
    env=gym.make('CartPole-v1')
    agent=Agent(alpha=0.0001,beta=0.0005,input_dims=[4],gamma=0.99,n_ouputs=2)
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
        
            agent.learn(observation,observation_new,reward,done)
            observation=observation_new
            score+=reward
        score_history.append(score)
        
    #filename='cartpole.png'
    #plotLearning(score_history,filename=filename,window=10)
    plt.plot([i for i in range(100)],score_history)
    plt.show()