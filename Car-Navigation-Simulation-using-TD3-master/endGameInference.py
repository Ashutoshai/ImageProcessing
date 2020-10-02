
# This file is used to visualize performance of trained models
#%%
import cv2
import numpy as np
import math

#from endGameEnv import car,carEndgameEnv # Main environment file used for training
from endGameEnvAllGoals import car,carEndgameEnv  # To test on all the goals
from endGameModels import TD3

import matplotlib.pyplot as plt
import torch

#%%
# Find out if GPU is available else use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
env_name = "carEndGameEnv" # Name of a environment
seed = 0 # Random seed number

#%%
env_name = "carEndGameEnv"
seed = 0

file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

#%% Define environment
car1 = car(carImgPath='images/car.png',size= (20,10),velocity=(2.0,0.0),angle=0.0)
env = carEndgameEnv(mapImgPath='images/citymap.png',maskImgPath='images/MASK1.png',carObj=car1)

#%%

torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.state_dim
action_dim = env.action_dim
max_action = env.max_action

# Create an instance of TD3 class
policy = TD3(state_dim, action_dim, max_action)
# Load trained model from './pytorch_models/' directory
policy.load(file_name, './pytorch_models/')

#%% Video settings

exampleImg = env.renderV2()
height, width = exampleImg.shape[:2]
size = (width,height)

out = cv2.VideoWriter('endgameSubmission.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

#%% Reward Plot

def plotMeanReward(meanReward):
    if(len(meanReward)>100):
        plt.figure(1)
        plt.clf()
        plt.title('Avg reward')
        plt.xlabel('Last n Iterations')
        plt.ylabel('Avg. Scores')
        plt.plot(meanReward[-100:])    
        plt.pause(0.01)  # pause a bit so that plots are updated

rewardBuffer = []
meanReward = []


#%%
liveView = True
printDebugValues = False
createVideo = True
reqdRewardPlot = False

# Reset environment
state = env.reset()

done = False
while done == False:    
    
    # Estimate action for current state
    action = policy.select_action(np.array(state))
    # Apply estimated action to the environment
    next_state,reward,done = env.step(action)
    # Get environment visualization image
    renderedImg = env.renderV2() 
    
    # If Debug information required
    if printDebugValues == True:
        print('action: ',action)    # print action estimated
        print('Reward: ', reward)   # print step reward
        print('stateValue: ',next_state[1:])    # print state values
        print(env.car)  # print car information

    # If live view of the environment visualization is required        
    if liveView == True:
        cv2.namedWindow('Live', cv2.WINDOW_NORMAL) 
        cv2.imshow("Live",renderedImg)  # Show image 
        if cv2.waitKey(1) == 27:    # Pause for window to display image. 
            break                   # If user enters 'ESC' break out of the loop
    
    # If video is required write the rendered image into video writer object
    if createVideo == True:
        out.write(renderedImg)

    # update the reward buffer        
    rewardBuffer.append(reward)
    # Maintain size of reward buffer to be 500
    if len(rewardBuffer) > 500:
        del rewardBuffer[0]
    
    # Estimate mean reward value          
    score = sum(rewardBuffer)/(len(rewardBuffer)+1.)
    meanReward.append(score)
    
    # If reward plot is required, plot it
    if reqdRewardPlot == True:
        plotMeanReward(meanReward)
    
    # Move over to next_state and repeat the loop
    state = next_state
    
      
cv2.destroyAllWindows() # destroy Live view window
out.release()       # Release the video writer

