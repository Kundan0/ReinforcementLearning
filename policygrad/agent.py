import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self,lr,input_dims,fc1_dims,fc2_dims,n_outputs):
        super(PolicyNetwork,self).__init__()
        self.lr=lr
        self.input_dims=input_dims
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.n_outputs=n_outputs
        self.fc1=nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2=nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3=nn.Linear(self.fc2_dims,self.n_outputs)
        self.optimizer=optim.Adam(self.parameters(),lr=self.lr)
        self.device=T.device( 'cuda' if T.cuda.is_available() else 'cpu' )
        self.to(self.device)

    def forward(self,observations):
        observations=T.Tensor(observations).to(self.device)
        x=F.relu(self.fc1(observations))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


class Agent(object):
    def __init__(self,lr,input_dims,n_actions,l1_size=128,l2_size=128,gamma=0.99):
        self.policy=PolicyNetwork(lr=lr,input_dims=input_dims,fc1_dims=l1_size,fc2_dims=l2_size,n_outputs=n_actions)
        self.reward_memory=[]
        self.action_memory=[]
        self.gamma=gamma

    def choose_action(self,observations):
        probabilities=F.softmax(self.policy.forward(observations),dim=0)
        action_probs=T.distributions.Categorical(probabilities)
        action=action_probs.sample()
        log_prob=action_probs.log_prob(action)
        self.action_memory.append(log_prob)
        return action.item()
    
    def store_rewards(self,reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        self.policy.optimizer.zero_grad()
        G=np.zeros_like(self.reward_memory,dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum=0
            discount=1
            for k in range(t,len(self.reward_memory)):
                G_sum+=self.reward_memory[k]*discount
                discount*=self.gamma
            G[t]=G_sum
        
        mean=np.mean(G)
        std=np.std(G)
        G=(G-mean)/std

        G=T.tensor(G,dtype=T.float).to(self.policy.device)
        
        loss=0
        
        for g,logprob in zip(G,self.action_memory):
            loss+=-g*logprob
            
            
        loss.backward()
        self.policy.optimizer.step()
        self.reward_memory=[]
        self.action_memory=[]





