# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:48:21 2020

@author: Gaurav
"""

#%%
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from collections import deque

import cv2 as cv
from PIL import Image
from scipy import ndimage
import copy
from PIL import Image as PILImage
import math

#%%
# Find out if GPU is available else use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

# Define ReplayBuffer class. It will store all the transitions as car is moving in the environment
class ReplayBuffer(object):
    # Take maximum size as 1e6. Replay buffer can store upto 1 Million transitions
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    # Define a method to add a transition into ReplayBuffer
    def add(self, transition):
        # Check if ReplayBuffer is full. If yes start from beginning.
        if len(self.storage) == self.max_size:
          self.storage[int(self.ptr)] = transition
          self.ptr = (self.ptr + 1) % self.max_size
        else:
          self.storage.append(transition)

    # Define a method to sample batch_size number of transitions from ReplayBuffer
    def sample(self, batch_size):
        # Take batch_size number of random indices
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        # Take empty lists corresponding to each of the elements
        batch_stateImgs,batch_stateValues, batch_next_stateImgs,batch_next_stateValues = [],[],[],[]
        batch_actions, batch_rewards, batch_dones = [],[],[]
        
        # Take out transitions corresponding to each of the random indices
        # and fill up the lists
        for i in ind:
              
            state, next_state, action, reward, done = self.storage[i]
            
            batch_stateImgs.append(np.array(state[0],copy=False))
            batch_stateValues.append(np.array(state[1:], copy=False))
            
            batch_next_stateImgs.append(np.array(next_state[0], copy=False))
            batch_next_stateValues.append(np.array(next_state[1:], copy=False))
            
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        
        # Return batches in the form of Numpy arrays    
        return np.array(batch_stateImgs),np.array(batch_stateValues),\
            np.array(batch_next_stateImgs),np.array(batch_next_stateValues),\
         np.array(batch_actions).reshape(-1, 1), np.array(batch_rewards).reshape(-1, 1),\
             np.array(batch_dones).reshape(-1, 1)
          

#%%

# Define Actor class
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # assuming input state image size to be 40x40
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(8),nn.Dropout2d(0.01)) # output_size = 38
          
        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(12),nn.Dropout2d(0.08)) # output_size = 36
        
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
              nn.ReLU())    # output_size = 36
        
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 18
        
        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(12),nn.Dropout2d(0.08)) # output_size = 16
                
        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout2d(0.08)) # output_size = 14
          
        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False)) # output_size = 12
                
        self.GAP = nn.Sequential(nn.AdaptiveAvgPool2d((1,1))) # output_size = 32
          
        # we will have 32 values coming from GAP layer and (state_dim-1) other state values 
        self.fc1 = nn.Linear(state_dim - 1 + 32, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
          
        self.max_action = max_action
    
    def forward(self, stateImg,stateValues):
        
        # Pass state image through convolution part of the model
        x = stateImg  
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.GAP(x)
        x = x.view(-1, 32)
        # concatenate with rest of the state elements and pass through fully connected layers
        x = torch.cat([x, stateValues], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # normalize output to lie in [-max_action,max_action]
        x = self.max_action * torch.tanh(self.fc3(x))   # tanh output is in [-1.0,1.0]
        return x

#%% Critic Model

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

    # Defining the first Critic CNN based network

        self.convblock1_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(8),nn.Dropout2d(0.01)) # output_size = 38
          
        self.convblock1_2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(12),nn.Dropout2d(0.08)) # output_size = 36
        
        self.convblock1_3 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
              nn.ReLU())    # output_size = 36
        
        self.pool1_1 = nn.MaxPool2d(2, 2) # output_size = 18
        
        self.convblock1_4 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(12),nn.Dropout2d(0.08)) # putput_size = 16
                
        self.convblock1_5 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout2d(0.08)) # output_size = 14
          
        self.convblock1_6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False)) # output_size = 12
        
        self.GAP1_1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))) # output_size = 32

    
    # we will have 32 values coming from GAP layer, (state_dim-1) other state values and action_dim number of action values
        self.fc1_1 = nn.Linear(state_dim - 1 + 32 + action_dim, 400)
        self.fc1_2 = nn.Linear(400, 300)
        self.fc1_3 = nn.Linear(300, 1)

    # Defining the second Critic neural network

        self.convblock2_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(8),nn.Dropout2d(0.01)) # output_size = 38
          
        self.convblock2_2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(12),nn.Dropout2d(0.08)) # output_size = 36
        
        self.convblock2_3 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
              nn.ReLU()) # output_size = 36
        
        self.pool2_1 = nn.MaxPool2d(2, 2) # output_size = 18
        
        self.convblock2_4 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(12),nn.Dropout2d(0.08)) # putput_size = 16
                
        self.convblock2_5 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout2d(0.08)) # output_size = 14
          
        self.convblock2_6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False)) # output_size = 12
        
        self.GAP2_1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))) # output_size = 32

    # we will have 32 values coming from GAP layer, (state_dim-1) other state values and action_dim number of action values

        self.fc2_1 = nn.Linear(state_dim - 1 + 32 + action_dim, 400)
        self.fc2_2 = nn.Linear(400, 300)
        self.fc2_3 = nn.Linear(300, 1)

    def forward(self, stateImg, stateValues, action):
    
    # Critic 1 forward pass
    # Pass state image through convolution part of the model
        x1 = self.convblock1_1(stateImg)
        x1 = self.convblock1_2(x1)
        x1 = self.convblock1_3(x1)
        x1 = self.pool1_1(x1)
        x1 = self.convblock1_4(x1)
        x1 = self.convblock1_5(x1)
        x1 = self.convblock1_6(x1)
        x1 = self.GAP1_1(x1)
        x1 = x1.view(-1, 32)

    # concatenate with rest of the state elements and action and pass through fully connected layers
        x1 = torch.cat([x1, stateValues, action], 1)
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)

    # Critic 1 forward pass
    # Pass state image through convolution part of the model
        x2 = self.convblock2_1(stateImg)
        x2 = self.convblock2_2(x2)
        x2 = self.convblock2_3(x2)
        x2 = self.pool2_1(x2)
        x2 = self.convblock2_4(x2)
        x2 = self.convblock2_5(x2)
        x2 = self.convblock2_6(x2)
        x2 = self.GAP2_1(x2)
        x2 = x2.view(-1, 32)

    # concatenate with rest of the state elements and action and pass through fully connected layers
        x2 = torch.cat([x2, stateValues, action], 1)
        x2 = F.relu(self.fc2_1(x2))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc2_3(x2)

        return (x1, x2)

    def Q1(self, stateImg, stateValues, action):

    # Critic 1 forward pass
    # Pass state image through convolution part of the model
        x1 = self.convblock1_1(stateImg)
        x1 = self.convblock1_2(x1)
        x1 = self.convblock1_3(x1)
        x1 = self.pool1_1(x1)
        x1 = self.convblock1_4(x1)
        x1 = self.convblock1_5(x1)
        x1 = self.convblock1_6(x1)
        x1 = self.GAP1_1(x1)
        x1 = x1.view(-1, 32)

    # concatenate with rest of the state elements and action and pass through fully connected layers

        x1 = torch.cat([x1, stateValues, action], 1)
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)

        return x1


#%%

# Define TD3 class. This class will have entire TD3 algorithm built into it
class TD3(object):
    
    # TD3 Constructor
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)    # Instance of Actor Model
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device) # Instance of Actor Target
        self.actor_target.load_state_dict(self.actor.state_dict())  # Copy weights of Actor Model into Actor Target
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=0.0001)  # Optimizer for training Actor network
        self.actor_lr_scheduler = StepLR(self.actor_optimizer, step_size=10000, gamma=0.9)  # Use StepLR to adjust Learning rate during training
        self.critic = Critic(state_dim, action_dim).to(device)  # Instance of Model Critics
        self.critic_target = Critic(state_dim, action_dim).to(device)   # Instance of Target Critics
        self.critic_target.load_state_dict(self.critic.state_dict())    # Copy weights of Model Critics into Target Citics
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=0.0001)    # Optimizer for training Critic networks
        self.critic_lr_scheduler = StepLR(self.critic_optimizer, step_size=10000, gamma=0.9)    # Use StepLR to adjust Learning rate during training
        self.max_action = max_action    # Maximum value of action

    # Define a method to estimate an action for given state
    def select_action(self, state):
        # first state element is cropped image
        stateImg = np.expand_dims(state[0],0)
        # Convert stateImg to tensor
        stateImg = torch.Tensor(stateImg).to(device)
        # Rest of the elements are float values. Create a float32 numpy array for it
        stateValues = np.array(state[1:], dtype=np.float32)
        # Convert StateValues to tensor with 1 Row 
        stateValues = torch.Tensor(stateValues.reshape(1, -1)).to(device)
        # Set model mode to 'evaluation' so batchNorm, DropOuts must be adjusted accordingly
        self.actor.eval()
        # Pass state through Actor Model forward pass to predict an action.
        # predicted action from tensor to numpy array before returning
        return(self.actor(stateImg,stateValues).cpu().data.numpy().flatten())
    

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        # Keep track of Critic and Actor Loss values
        criticLossAvg = 0.0
        actorLossAvg = 0.0
        for it in range(iterations):
            
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_stateImgs,batch_stateValues, batch_next_stateImgs,batch_next_stateValues,\
                batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      
            stateImg = torch.Tensor(batch_stateImgs).to(device)
            stateValues = torch.Tensor(batch_stateValues).to(device)
            next_stateImgs = torch.Tensor(batch_next_stateImgs).to(device)
            next_stateValues = torch.Tensor(batch_next_stateValues).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            
            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_stateImgs,next_stateValues)
            
            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_stateImgs,next_stateValues, next_action)
            
            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()
            
            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(stateImg,stateValues,action)
            
            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            #print(critic_loss.item(),type(critic_loss.item()))
        
            criticLossAvg += critic_loss.item()
            
            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic.train() # Set Model mode to 'training'
            self.critic_optimizer.zero_grad()   # reset grad values
            critic_loss.backward()              # update grad values
            self.critic_optimizer.step()        # update model parameters
            self.critic_lr_scheduler.step()     # step through critic LR scheduler
            
            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(stateImg,stateValues, self.actor(stateImg,stateValues)).mean()
                actorLossAvg += actor_loss.item()
                self.actor.train()  # Set Model mode to 'training'
                self.actor_optimizer.zero_grad()    # Reset grad values
                actor_loss.backward()               # update grad values
                self.actor_optimizer.step()         # update model parameters
                self.actor_lr_scheduler.step()      # step through actor LR scheduler
                
                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                  target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                  target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                  
        criticLossAvg /= iterations 
        actorLossAvg /= iterations       
        print('Avg CriticLoss: ',criticLossAvg,' Avg ActorLoss ',actorLossAvg, \
              ' ActorLR: ',self.actor_lr_scheduler.get_last_lr(),' CriticLR: ',self.critic_lr_scheduler.get_last_lr())
  
    def save(self, filename, directory):
          torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
          torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
    def load(self, filename, directory):
          self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename),map_location=torch.device('cpu')))
          self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename),map_location=torch.device('cpu')))
          
          
#policy = TD3(4, 1, 5.0)          