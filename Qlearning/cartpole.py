from agent import Agent
import gym
import matplotlib.pyplot as plt
 
if __name__=='__main__':
    env=gym.make('CartPole-v1')
    agent=Agent(input_dims=[4],n_actions=2)
    score_history=[]
    score=0
    n_episodes=2500
    for i in range(n_episodes):
        print('episode',i,'score %.3f' % score)
        done=False
        score=0
        observation=env.reset()
        
        while not done:
            #env.render()
            action=agent.choose_action(observation)
            
            observation_new,reward,done,info=env.step(action)
        
            agent.learn(observation,observation_new,action,reward)
            observation=observation_new
            score+=reward
        score_history.append(score)
        
   
    plt.plot([i for i in range(n_episodes)],score_history)
    plt.show()