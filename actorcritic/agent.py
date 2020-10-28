import torch as t 
import torch.nn as nn
import torch.nn.functional as f 
import torch.optim as optim

class Network(nn.Module):
    def __init__(self,lr,input_dims,n_ouputs,fc1_dims,fc2_dims):
        super(Network,self).__init__()
        self.lr=lr
        self.input_dims=input_dims
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.n_outputs=n_ouputs
        self.fc1=nn.Linear(*input_dims,fc1_dims)
        self.fc2=nn.Linear(fc1_dims,fc2_dims)
        self.fc3=nn.Linear(fc2_dims,n_ouputs)
        self.optimizer=optim.Adam(self.parameters(),lr=self.lr)

    def forward(self,state):
        
        state=t.tensor(state,dtype=t.float32)
    
        x=f.relu(self.fc1(state))
        
        x=f.relu(self.fc2(x))
        x=self.fc3(x)
        
        return x

class Agent(object):
    def __init__(self,alpha,beta,input_dims,gamma,n_ouputs=2,fc1_dims=32,fc2_dims=32):
        self.actor=Network(alpha,input_dims,n_ouputs,fc1_dims,fc2_dims)
        self.critic=Network(beta,input_dims,1,fc1_dims,fc2_dims)
        self.log_prob=None
        self.gamma=gamma

    def choose_action(self,observations):
        
        probabilities=f.softmax(self.actor.forward(observations))
        
        actions_probs=t.distributions.Categorical(probabilities)
        action=actions_probs.sample()
        
        self.log_prob=actions_probs.log_prob(action)
        
        return action.item()
    
    def learn(self,state,next_state,reward,done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        critic_value=self.critic.forward(state)
        critic_value_=self.critic.forward(next_state)
        delta=(reward+self.gamma*critic_value_*(1-int(done)))-critic_value
        actor_loss=(-self.log_prob)*delta
        critic_loss=delta**2
        (actor_loss+critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()


        