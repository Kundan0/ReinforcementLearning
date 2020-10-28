import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np 
class Qnetwork(nn.Module):
    def __init__(self,lr,input_dims,n_outputs,hidden_layer=256):
        super(Qnetwork,self).__init__()
        self.lr=lr
        self.input_dims=input_dims
        self.n_outputs=n_outputs
        self.hidden_layer=hidden_layer
        
        self.fc1=nn.Linear(*input_dims,hidden_layer)
        self.fc2=nn.Linear(hidden_layer,hidden_layer)
        self.fc3=nn.Linear(hidden_layer,n_outputs)
        self.loss=nn.MSELoss()
        self.optimizer=optim.Adam(self.parameters(),lr=self.lr)
        self.device=t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        
        
    def forward(self,observations):
        actions=f.relu(self.fc1(t.tensor(observations).to(self.device)))
        actions=f.relu(self.fc2(actions))
        actions=f.softmax(self.fc3(actions))
        return  actions
    
class Agent(object):
    def __init__(self,input_dims,n_actions,epsilon=1.0,eps_min=0.01,eps_dec=1e-05,lr=0.0001,gamma=0.99):
        self.Qnet=Qnetwork(lr,input_dims,n_actions)
        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_min=eps_min
        self.eps_dec=eps_dec
        self.action_space=[i for i in range(n_actions)]
    def decrement_epsilon(self):

        if self.epsilon>self.eps_min:
            self.epsilon-=self.eps_dec
        else:
            self.epsilon=self.eps_min

    def choose_action(self,observations):
        observation=t.tensor(observations,dtype=t.float).to(self.Qnet.device)
        actions=self.Qnet.forward(observation)
        action=t.argmax(actions).item()
        return action
        
    def learn(self,state,state_,action,reward):
        self.Qnet.optimizer.zero_grad()
        states=t.tensor(state,dtype=t.float).to(self.Qnet.device)
        states_=t.tensor(state_,dtype=t.float).to(self.Qnet.device)
        rewards=t.tensor(reward).to(self.Qnet.device)
        actions=t.tensor(action).to(self.Qnet.device)
        q_pred=self.Qnet.forward(states)[actions]
        q_next=self.Qnet.forward(states_).max()
        q_target=rewards+self.gamma*q_next
        loss=self.Qnet.loss(q_target,q_pred).to(self.Qnet.device)
        loss.backward()
        self.Qnet.optimizer.step()
        self.decrement_epsilon()
    



