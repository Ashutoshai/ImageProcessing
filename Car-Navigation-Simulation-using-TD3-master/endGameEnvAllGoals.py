

# This file is similar to endGameEnv only difference is it doesn't end episode 
# on max episode steps or with achievement of 3 goals. It goes on till all the goals
# in the goal list have been achieved
#%%
import cv2
import numpy as np
import copy
import random
import math
from scipy import ndimage
from endGameUtilities import mergeImages,imgRotate


#%%
class car(object):
    
    def __init__(self,carImgPath,size=(0,0),velocity=(5.0,0.0),angle=0.0):
        
        """
        Parameters
        ----------
        carImgPath : str
            Path of image to be used as car
        size : Tuple (Width,Height), optional
            Car image will be resized to this value. The default is (0,0) which means take image as it is.
        velocity : Tuple (VelocityX,VelocityY), optional
            Car's initial velocity will be set to this value. The default is (5.0,0.0).
        angle : float, optional
            Car's initial angle of inclination will be set to this value. Angle of inclination
            is measured wrt x axis such that anticlockwise direction is positive.The default is 0.0.
            angle is represented in degrees

        Returns
        -------
        None.

        Description
        -------
        This is constructor for class 'car'
        """
        
        self.imgPath = carImgPath
        self.img =  cv2.imread(self.imgPath)
        # if given size is default value or same as image size, no need to resize image
        if size == (0,0) or ((size[1],size[0]) == self.img.shape[:2]):
            (self.height,self.width) = self.img.shape[:2]
        else : # resize image according to provided size
            self.img = cv2.resize(src=self.img, dsize=size) # cv2.resize takes dsize in order (width,height)
                                                            # while .shape gives size in order (row,col) i.e. (height,width)
            (self.width,self.height) = size
        
        # Set initial Car velocity
        (self.velocityX,self.velocityY) = velocity
        
        # Set initial Car angle
        self.angle = angle
        # Set initial Car position
        # It must be noted that Car's location is represent in a coordinate frame such that bottom left 
        # corner represents origin unlike image or numpy array coordinate from  which has origin at top left
        self.posX = 0
        self.posY = 0
        
    # Printing car information. When Car object is passed to print this method will be called
    def __str__(self): 
        return 'imgPath:{}, width:{}, height:{}, velocityX:{}, velocityY:{},angle:{}, posX:{}, posY:{}'\
            .format(self.imgPath,self.width,self.height,self.velocityX,self.velocityY,self.angle,self.posX,self.posY)
        
    def move(self,rotation=0.0):
        """
        Parameters
        ----------
        rotation : float, optional
            This represents rotation to be applied to Car. The default is 0.0.

        Returns
        -------
        None.

        Description
        -------
        This method will update the Car angle by 'rotation' angle and 
        move the car by one step along its angle of inclination as per its velocity
            
        """
        # Update Car angle
        self.angle = (self.angle + rotation)
        
        # adjust Car angle to range [-180,180]
        if (self.angle < -180.0) :
            self.angle = self.angle + 360.0
        elif self.angle > 180.0:
            self.angle = self.angle - 360.0
        
        # convert angle from degrees to radians as math trigonometric functions accept radians
        angle = math.radians(self.angle)
        
        # compute displacement along x and y axis based on Car angle and velocity
        dx = (self.velocityX * math.cos(angle)) - (self.velocityY * math.sin(angle))
        dy = (self.velocityY * math.cos(angle)) + (self.velocityX * math.sin(angle))
        
        # Apply displacements to get new car location
        self.posX = int(self.posX + dx)
        self.posY = int(self.posY + dy)
        
        
    def getCarImg(self):
        """
        This method returns car image rotated by current car angle.
        """
        # Image Rotation about its ceter keeping entire view of image intact
        # has been taken from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        # Github location https://github.com/jrosebr1/imutils
        
        h = self.height
        w = self.width
        
        # Compute center of car image
        (cX, cY) = (w / 2, h / 2)
    
        # grab the rotation matrix then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
    
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    
        # perform the actual rotation and return the image
        return cv2.warpAffine(self.img, M, (nW, nH))
    
    def setPosition(self,posX = 0,posY = 0):
        """
        Parameters
        ----------
        posX : int, optional
            X coordinate of car's location. The default is 0.
        posY : TYPE, optional
            Y coordinate of car's Location. The default is 0.

        Returns
        -------
        None.
        
        Description
        -----------
        This method sets Car's current location to provided values
        """
        self.posX = posX
        self.posY = posY
    
    
    def setAngle(self,angle = 0.0):
        """
        Parameters
        ----------
        angle : float, optional
            Car's angle. The default is 0.0.

        Returns
        -------
        None.

        Description
        -----------
        This method sets Car's current inclination to provided angle
        """
        self.angle = angle
        
    def setVelocity(self, velX,velY):
        """
        Parameters
        ----------
        velX : float
            velocity along x axis
        velY : float
            velocity alongg y axis

        Returns
        -------
        None.
        
        Description
        -----------
        This method sets Car's current velocity to provided values
        
        """
        self.velocityX = velX
        self.velocityY = velY
            

#%%
# Main Car navigation environment class. Instances of this class will be used
# to train on and play Car navigation game
class carEndgameEnv(object):
    
    def __init__(self,mapImgPath,maskImgPath,carObj):
        """
        Parameters
        ----------
        mapImgPath : str
            Path of image to be used as map canvas. This will be used for visualization 
            of car running on a map. This will not be used for any calculations
        maskImgPath : str
            Path of image to be used as map mask indicating road and non-road (sand) 
            locations. This image will be used for internal computations. The code assumes that
            pixel value '0' represents Road and '1' represents Non-Road (Sand).
        carObj : object of class car
            This instance of Car class will be used to keep track of Car's location and also
            to generate Car's visual on map canvas

        Returns
        -------
        None.
        
        Description
        -----------
        This method is a constructor of class carEndGame. It prepares an instance of the 
        class with necessary initializations
        """
        
        # Load map image. Thisis a colored image. It will be used only for visualization
        # of car running on map canvas
        self.mapImgPath = mapImgPath
        self.mapImg = cv2.imread(self.mapImgPath)
        (self.mapHeight,self.mapWidth) = self.mapImg.shape[:2]
        
        # load mask image in grayscale mode (0 = Road, 1 = Sand)
        self.maskImgPath = maskImgPath
        self.maskImg = cv2.imread(self.maskImgPath,cv2.IMREAD_GRAYSCALE)
        
        # Car object
        self.car = carObj
        
        # Set the Car reset location such that Car is on the road and pointing along the road
        self.resetLoc = [188,457,10]
        self.car.setPosition(int(self.resetLoc[0]),int(self.resetLoc[1]))
        # set car angle as float32 numpy array because later output action of Actor class
        # will be in Numpy float32 array form so it will be easier to manage
        self.car.setAngle(np.array([self.resetLoc[2]],dtype = np.float32).flatten())
        
        
        # Create a list of Goals. During any episode 3 goals will be picked up from this list at random
        self.goalList = [[260,476],[885,568],[501,608],[346,426],[611,29],\
                         [1130,236],[472,335],[1415,626],[150,200],[700,583]]    
        
        # self.goalList = [[1170,612],[260,476],[599,147],[885,568],[501,608],\
        #                  [346,426],[1140,429],[611,29],[45,257],[228,628]]
        
        # Shuffle the goal list and select first point as goal
        #random.shuffle(self.goalList)
        self.goalX = self.goalList[0][0]
        self.goalY = self.goalList[0][1]
        self.goalIndex = 0
        self.numGoals = len(self.goalList)
        
        # swap flag will be used to keep track of number of goals achieved so far and switching the goals
        self.swap = 0
        
        #### State space
        # It has 4 elements
        # 0: Crop of mask image centered at car's location rotated by (90-car.angle).This corresponds to
        #       front view of the car. 
        # 1: Distance to Goal scaled by maxGoalDistance
        # 2: Orientation of Goal wrt car 
        # 3: Negative orientation of Goal wrt car
        
        self.state = self.prevState = [[],0.0,0.0,0.0]
        self.state_dim = len(self.state)
        
        # cropping parameters
        self.cropSize = 40
        
        # arrow points
        self.arrowSize = 4
        aSz = int(self.arrowSize)
        arrowPts = np.array([[(aSz*4/2-2*aSz,aSz*4/2+aSz),(aSz*4/2+0,aSz*4/2+aSz),\
                     (aSz*4/2+0,aSz*4/2+2*aSz),(aSz*4/2+2*aSz,aSz*4/2+0),\
                         (aSz*4/2+0,aSz*4/2-2*aSz),(aSz*4/2+0,aSz*4/2-aSz),\
                             (aSz*4/2-2*aSz,aSz*4/2-aSz)]],dtype=np.int32)
        # get empty image of size greater than expected arrow size
        self.arrowImg = np.zeros((aSz*4,aSz*4),dtype=np.uint8)
        
        # Generate Arrow image with gray scale pixel value of 127.
        # This image will be pasted in the crop of road mask to represent car. 
        # Arrow image creates assymmetry helping DNN make sense of
        # which way car is pointing
        cv2.fillPoly(self.arrowImg,arrowPts,127)
        
    
        # Calculate distance to Goal from current car location
        self.goalDistance = np.sqrt((self.car.posX - self.goalX)**2 + (self.car.posY - self.goalY)**2)
        self.prevGoalDistance = self.goalDistance
        
        # Calculate maximum euclidean distance possible (Diagonal of the rectangle). Goal distances will be 
        # scaled by maxGoalDistance to use as one of the state values.
        self.maxGoalDistance = np.sqrt((0 - self.mapWidth-1)**2 + (0 - self.mapHeight-1)**2)
        
                
        # Action space 
        # we have just one action 'car rotation'. car will rotate by this value
        self.action_dim = 1
        self.max_action = 5.0 # +/- max_action is max rotation possible at a time 
        
        # rewards
        self.prevReward = self.reward = 0.0
        # Keep track total episode reward
        self.episodeReward = 0.0
        
        #Keeping track of how long the car has been stuck at boundaries
        self.stuckAtBoundaryCount = 0
        # If stuck at boundary for more than maxStuckAtBoundaryCount end the episode
        self.maxStuckAtBoundaryCount = 50
        
        # tracking how many steps car ran on sand or road during an episode
        self.sandCount = 0
        self.roadCount = 0
        # tracking how many steps car was at the boundaries
        self.boundaryCount =0
        
        # Keep track of number of steps taken in an episode
        self.episodeSteps = 0
        # max number of episodes
        self._max_episode_steps = 5000  # This value was set based on observations
        
    def reset(self):    
        """
        Returns
        -------
        Type: list 
            This method returns a list representing state of the environment 
            after reset()
            
        Description
        -----------
        This method resets the environment to its initial state. This method must
        be called at the end of an episode if environment needs to be reset.
        Environment itself will not call this function

        """
        # Reset reward values, 
        self.reward = self.prevReward = 0.0
        self.episodeReward = 0.0
        # Reset episode steps counter
        self.episodeSteps = 0
        
        # Everytime reset car position to same location
        self.car.setPosition(int(self.resetLoc[0]),int(self.resetLoc[1]))
        # Reset car inclination angle
        self.car.setAngle(np.array([self.resetLoc[2]],dtype = np.float32).flatten())
        # Reset car velocity
        self.car.setVelocity(2.0, 0)            
            
        # Make done = False at the beginning of an episode
        self.done = False
        
        
        # Reset various counters
        self.stuckAtBoundaryCount = 0
        self.sandCount = 0
        self.roadCount = 0
        self.boundaryCount =0
        
        # Shuffle the goalList and take the first point as goal
        self.goalX = self.goalList[0][0]
        self.goalY = self.goalList[0][1]
        self.goalIndex = 0
        self.numGoals = len(self.goalList)
        self.swap = 0
        
        # Calculate goal Distance 
        self.goalDistance = np.sqrt((self.car.posX - self.goalX)**2 + (self.car.posY - self.goalY)**2)
        self.prevGoalDistance = self.goalDistance
        
        # Estimate current environment state
        self.state = self.prevState = self.estimateStateV2()
        return self.state
        
    # here action is angle of rotation
    def step(self, action):
        """
        Parameters
        ----------
        action : np.float32 type numpy array
            Action to be taken.

        Returns
        -------
        state (list)
            List representing state of the environment after taking
            given action.
        reward (float)
            It is reward obtained for taking given action in current state.
        done (boolean)
            It indicates whether episode ended with this action

        Description
        -----------
        This method applies given action to the Car and calculates resulting state,
        reward and whether episode ended with this step
        """
        
        # Save Car's location before updating it
        prevCarPosX = self.car.posX
        prevCarPosY =self.car.posY
        
        # Move the car 
        self.car.move(action)
        
        # check if car is inside map else bring it in
        if self.car.posX < 0: self.car.posX = 0
        elif self.car.posX > (self.mapWidth-1): self.car.posX = int(self.mapWidth - 1)
        
        if self.car.posY < 0: self.car.posY = 0
        elif self.car.posY > (self.mapHeight-1): self.car.posY = int(self.mapHeight - 1)
        
        
        # Reset reward for current step
        self.reward = 0.0
        self.done = False
        
        
        # calculate goal Distance    
        self.prevGoalDistance = self.goalDistance       # save previous value of goal distance
        self.goalDistance = np.sqrt((self.car.posX - self.goalX)**2 + (self.car.posY - self.goalY)**2)
               
        # get car coordinates in image reference frame 
        carX,carY = self.getCarCoordinates()
        
        # check if car is on sand
        if self.maskImg[int(carY),int(carX)] > 0:
            self.sandCount += 1   # Car is on Sand in this step so increment sand count 
            self.car.setVelocity(1.5, 0)    # Reduce Car velocity
            self.reward = -5.0      # bad step penalize
        
        else: # car is on road
            self.roadCount += 1   # Car is on Road in this step so increment Road count 
            self.car.setVelocity(2.0, 0)    # Increase Car velocity
            # if moving towards goal
            if self.goalDistance < self.prevGoalDistance:
                self.reward = 0.5           # Give incentive with positive reward
            else:   # else give living penalty
                self.reward = -0.9          # Living penalty
            
        # Check if car is at the boundary (It will not cross boundary because we made sure after car.move())  
        if((self.car.posX <= 0 or self.car.posX >= self.mapWidth-1)  or 
            (self.car.posY <= 0 or self.car.posY >= self.mapHeight-1) ) :
            self.boundaryCount += 1 
            self.reward -= 10 
           
        # Check if Car is near current goal then give incentive and switch the Goal   
        if self.goalDistance < 30:
            print('Goal:(',self.goalX,self.goalY,')')
            self.reward += 50
            self.goalIndex += 1
            if self.goalIndex == self.numGoals:
                self.done = True
                print('Episode Done:All goals achieved')# end the episode if all the 3 goals are achieved
            
            self.goalIndex %= self.numGoals
            self.goalX = self.goalList[self.goalIndex][0]
            self.goalY = self.goalList[self.goalIndex][1]
        
        # Add current step reward to total episode reward
        self.episodeReward += self.reward
        # Increment Episode steps count
        self.episodeSteps += 1
        
        # If episode has not ended because of hitting all goals check other conditions
        if self.done == False: 
            
            # Check if maximum episode steps have been exhausted
            #if self.episodeSteps >= self._max_episode_steps:
            #    self.done = True
            #    print('Episode Done:Max episode steps exhausted')
            
            # Check if car is stuck at the boundaries i.e. car is at the boundary and not moving
            if((self.car.posX <= 0 or self.car.posX >= self.mapWidth-1)  or 
                (self.car.posY <= 0 or self.car.posY >= self.mapHeight-1) ): 
                if (prevCarPosX-self.car.posX == 0 and prevCarPosY-self.car.posY == 0): # Car stuck
                    self.stuckAtBoundaryCount += 1
                else:
                    self.stuckAtBoundaryCount = 0   # reset count to 0 if not consecutive stucks
            # If the car is stuck to boundaries for too long end the episode
            if self.stuckAtBoundaryCount >= self.maxStuckAtBoundaryCount:
                self.done = True
                print('Episode Done:Car stuck')
        
        # If episode has ended summerise counts of steps taken by Car
        if self.done == True:
            print('Episode steps:',self.episodeSteps,' Sand count: ',self.sandCount,' Road Count: ',self.roadCount)
        
        # Save current state
        self.prevState = self.state
        # Estimate new state after having taken the current step
        self.state = self.estimateStateV2()
        
        # return state,reward and done flag for current step
        return self.state,self.reward,self.done
    
    
    def estimateState(self):
        """
        Returns
        -------
        state : list
        
        Description
        -----------
        Estimate current state of the environment. 
        State consists of 4 elements
        # 0: Crop of mask image centered at car's location with Arrow image oriented at Car's angle
                pasted on it
        # 1: Distance to Goal scaled by maxGoalDistance
        # 2: Orientation of Goal wrt car 
        # 3: Negative orientation of Goal wrt car

        """
        
        ################ State Element 0 #################################

        # Fetch the crop of Mask image of size 'crop_size' centered at car's location
        cropWithCar = self.getCurrentMaskCrop()
        w,h =cropWithCar.shape[:2]
        
        # Rotate the image of an Arrow by Car's current inclination angle
        arrowImgRotated = imgRotate(self.arrowImg,self.car.angle)
        
        # Check the size of rotated image
        arrowH,arrowW = arrowImgRotated.shape[:2]
        
        # Calculate where in the cropped image , arrow image's top left corner should be
        plotX = int(w/2 - arrowW/2)
        plotY = int(h/2 - arrowH/2)
        
        # Blend Arrow image with portion of cropped image at the location calculated above
        # mergeImages combines two images in proportion alpha and beta (output = alpha*img1 + beta*img2)
        blendedPart = mergeImages(img1=arrowImgRotated,alpha=1.0,img2=cropWithCar[plotY:plotY+ arrowH,plotX:plotX+arrowW ],beta=0.0)
       
        # paste the blended part onto the cropped image at location calculated earlier
        cropWithCar[plotY:plotY+ arrowH,plotX:plotX+arrowW ]  = blendedPart
        
        # Normalize the crop image to [0.0,1.0] pixel values. This image will be used as one of the states
        # to be fed into the CNN. So it is better to provide normalized image rather than [0,255] image
        cropWithCar = (cropWithCar)/(255.0)   # normalize image to lie between [0,1.0]
        
        # output dimension should be 1xHxW so expand dimension
        cropWithCar = np.expand_dims(cropWithCar,0)
        
        
        ####################### State element 1 ###############################
        
        # calculate distance to goal
        goalDistance = np.sqrt((self.car.posX - self.goalX)**2 + (self.car.posY - self.goalY)**2)
        # normalize goalDistance by maxGoalDistance to lie between [0.0,1.0]
        goalDistance = (goalDistance)/(self.maxGoalDistance)
        
        ####################### State element 2 and 3 ###############################
        
        # Estimate vector joining car position and goal position
        xx = self.goalX - self.car.posX
        yy = self.goalY - self.car.posY
        angle = math.radians(self.car.angle)
        
        # code taken from kivy vector.rotate (https://github.com/kivy/kivy/blob/420c8e2b8432d5363ed0662be7a649c2a8b86274/kivy/vector.py#L289)

        # Estimate vector representing Car's pointing direction
        carVecX = (self.car.velocityX * math.cos(angle)) - (self.car.velocityY * math.sin(angle))
        carVecY = (self.car.velocityY * math.cos(angle)) + (self.car.velocityX * math.sin(angle))
        
        # code taken from kivy vector.angle (https://github.com/kivy/kivy/blob/420c8e2b8432d5363ed0662be7a649c2a8b86274/kivy/vector.py#L289)
       
        # Estimate angle between vectors calculated above 
        orientation = -(180 / math.pi) * math.atan2(
            carVecX * yy - carVecY * xx,
            carVecX * xx + carVecY * yy)
        
        # normalize orientation to range [-1.0,1.0]
        orientation /= 180
        
        # Assemble state List
        state = [cropWithCar,goalDistance,-orientation,orientation]
        
        return state
    
    def estimateStateV2(self):
        """
        Returns
        -------
        state : list
        
        Description
        -----------
        Estimate current state of the environment. 
        State consists of 4 elements
        # 0: Crop of mask image centered at car's location rotated by (90 - Car.angle) and
                arrow image pasted on it.
        # 1: Distance to Goal scaled by maxGoalDistance
        # 2: Orientation of Goal wrt car 
        # 3: Negative orientation of Goal wrt car

        """
        
        ################ State Element 0 #################################

        # Fetch the crop of Mask image of size 'crop_size' centered at car's location and rotated
        # by (90 - car.angle). This represents car's front view
        cropWithCar = self.getCurrentMaskCropV2()
        w,h =cropWithCar.shape[:2]
        
        # Rotate the image of an Arrow by 90 because car will always be facing up in
        # its front view
        arrowImgRotated = imgRotate(self.arrowImg,90.0) # arrow always facing up
        
        # Check the size of rotated image
        arrowH,arrowW = arrowImgRotated.shape[:2]
        
        # Calculate where in the cropped image , arrow image's top left corner should be
        plotX = int(w/2 - arrowW/2)
        plotY = int(h/2 - arrowH/2)
        
        
        # Blend Arrow image with portion of cropped image at the location calculated above
        # mergeImages combines two images in proportion alpha and beta (output = alpha*img1 + beta*img2)
        blendedPart = mergeImages(img1=arrowImgRotated,alpha=1.0,img2=cropWithCar[plotY:plotY+ arrowH,plotX:plotX+arrowW ],beta=0.0)
        
        # paste the blended part onto the cropped image at location calculated earlier
        cropWithCar[plotY:plotY+ arrowH,plotX:plotX+arrowW ]  = blendedPart
        
        # Normalize the crop image to [0.0,1.0] pixel values. This image will be used as one of the states
        # to be fed into the CNN. So it is better to provide normalized image rather than [0,255] image
        cropWithCar = (cropWithCar)/(255.0)   
        
        # output dimension should be 1xHxW so expand dimension
        cropWithCar = np.expand_dims(cropWithCar,0)
        
        ####################### State element 1 ###############################
        
        # calculate distance to goal
        goalDistance = np.sqrt((self.car.posX - self.goalX)**2 + (self.car.posY - self.goalY)**2)
       
        # normalize goalDistance by maxGoalDistance to lie between [0.0,1.0]
        goalDistance = (goalDistance)/(self.maxGoalDistance)
        
        ####################### State element 2 and 3 ###############################
        
        # Estimate vector joining car position and goal position
        xx = self.goalX - self.car.posX
        yy = self.goalY - self.car.posY
        angle = math.radians(self.car.angle)
        
        # code taken from kivy vector.rotate (https://github.com/kivy/kivy/blob/420c8e2b8432d5363ed0662be7a649c2a8b86274/kivy/vector.py#L289)

        # Estimate vector representing Car's pointing direction
        carVecX = (self.car.velocityX * math.cos(angle)) - (self.car.velocityY * math.sin(angle))
        carVecY = (self.car.velocityY * math.cos(angle)) + (self.car.velocityX * math.sin(angle))
        
        # code taken from kivy vector.angle (https://github.com/kivy/kivy/blob/420c8e2b8432d5363ed0662be7a649c2a8b86274/kivy/vector.py#L289)
       
        # Estimate angle between vectors calculated above 
        orientation = -(180 / math.pi) * math.atan2(
            carVecX * yy - carVecY * xx,
            carVecX * xx + carVecY * yy)
        
        # normalize orientation to range [-1.0,1.0]
        orientation /= 180
        
        # Assemble the state list
        state = [cropWithCar,goalDistance,-orientation,orientation]
        
        return state
    
    
    def getCurrentMaskCrop(self):
        """
        Returns
        -------
        croppedImg : img
            
        Description
        -----------
        This method returns a crop of mask image centered at car's current location

        """
        
        # The Car could be at the boundary so we must pad mask before extracting cropped image
        border = int(np.max([self.cropSize/2,self.car.width/2,self.car.height/2]) + 5) # 5 safety margin
        # Apply padding
        paddedMask = cv2.copyMakeBorder(self.maskImg, border, border, border, border, cv2.BORDER_CONSTANT,value=255)
        
        # Fetch car coordinates in image reference frame
        coordX,coordY = self.getCarCoordinates()
        # Estimate top left corner of the crop while considering border padding
        cropXloc = int(coordX - self.cropSize/2 + border)
        cropYloc = int(coordY - self.cropSize/2 + border)
        
        # Crop the image from padded image        
        croppedImg = paddedMask[int(cropYloc) : int(cropYloc+self.cropSize),int(cropXloc) : int(cropXloc+self.cropSize)]
        return croppedImg
    
    def getCurrentMaskCropV2(self):
        """
        Returns
        -------
        croppedImg : img
            
        Description
        -----------
        This method returns a crop of mask image centered at car's current location and rotated by (90-car.angle)

        """
        
        # In this version we generate crop such that car is always facing front, crop reorients by -car angle
        
        # to do this we must first crop out image larger than crop size, rotate by (90-carAngle)
        # then crop required crop size
        
        # First crop of twice the size of required crop
        firstCropSize = self.cropSize*2
        
        # The Car could be at the boundary so we must pad mask before extracting cropped image
        border = int(np.max([firstCropSize/2,self.car.width/2,self.car.height/2]) + 10) # 10 safety margin
        
        # Apply padding
        paddedMask = cv2.copyMakeBorder(self.maskImg, border, border, border, border, cv2.BORDER_CONSTANT,value=255)
              
        # Fetch car coordinates in image reference frame
        coordX,coordY = self.getCarCoordinates()
        
        # Estimate top left corner of the crop while considering border padding
        cropXloc = int(coordX - firstCropSize/2 + border)
        cropYloc = int(coordY - firstCropSize/2 + border)
        
        # Crop the image from padded image     
        croppedImg = paddedMask[int(cropYloc) : int(cropYloc+firstCropSize),int(cropXloc) : int(cropXloc+firstCropSize)]
        
        
        # Rotate first crop by (90-car.angle)
        croppedImg = ndimage.rotate(croppedImg,(90-self.car.angle[0]))
        
        # second crop
        (h,w) = croppedImg.shape[:2]
        # estimate top left of the crop
        cropXloc = int(w/2 - self.cropSize/2)
        cropYloc = int(h/2 - self.cropSize/2)
        
        # Perform second crop
        croppedImg = croppedImg[int(cropYloc) : int(cropYloc+self.cropSize),int(cropXloc) : int(cropXloc+self.cropSize)]
                
        return croppedImg
        
    
    def sampleActionSpace(self):
        # This method returns a sample action value in range [-max_action,max_action]. Type of the value 
        # return is np.float32 numpy array
        return np.array(random.uniform(-1*self.max_action,self.max_action),dtype = np.float32).flatten()
    
    # render should return image with car plotted over map
    def render(self):
        """
        Returns
        -------
        img
        
        Description
        -----------
        This method returns an image of map canvas with car image plotted over it. This image
        can be used to visualize car motion on the map canvas

        """
        
        # Fetch the car image rotated by current car inclination angle
        carImg = self.car.getCarImg()
        # Get image's dimensions
        carHeight,carWidth = carImg.shape[:2]
        
        # get car coordinates in image reference frame
        centerX,centerY = self.getCarCoordinates()
        
        # Estimate location of top left corner of car image in map canvas
        plotX = int(centerX - carWidth/2)
        plotY = int(centerY - carHeight/2)
        #print('plotX',plotX,'plotY',plotY)

        # Make a copy of original map image so that changes arre not permanently 
        # written
        
        mapImg2 = copy.deepcopy(self.mapImg)
        
        # The car could be at the border so we add padding first, plot car and then crop out the border
        
        # max width and height that 10,20 car take is 22,22 so
        # we must have border of 11. but we set border as 15 as safe margin 
        border = 15
        mapImg2 = cv2.copyMakeBorder(mapImg2, border, border, border, border, cv2.BORDER_CONSTANT,value=0 )
        
        # take care of border for car image top left corner
        plotX += border
        plotY += border
        
        # Blend car image with portion of map image at the location calculated above
        # mergeImages combines two images in proportion alpha and beta (output = alpha*img1 + beta*img2)
        blendedPart = mergeImages(carImg,1.0,mapImg2[plotY:plotY+ carHeight,plotX:plotX+carWidth ],0.0)
        
        # paste the blended part onto map image
        mapImg2[plotY:plotY+ carHeight,plotX:plotX+carWidth ]  = blendedPart
        
        # Remove the borders
        mapImg2 = mapImg2[border:border+self.mapHeight,border:border+self.mapWidth]
        
        #plot goal circles
        #Inner solid circle
        mapImg2 = cv2.circle(mapImg2,(self.goalX,self.mapHeight-1-self.goalY),8,(255,0,0),-1)
        # Outer ring representing area of radius 30 around goal
        mapImg2 = cv2.circle(mapImg2,(self.goalX,self.mapHeight-1-self.goalY),30,(0,0,255),2)
        return mapImg2
     
    # render should return image with car plotted over map also state in bottom right
    def renderV2(self):
        """
        Returns
        -------
        img
        
        Description
        -----------
        This is another version of render() method.
        This method returns an image of map canvas with car image plotted over it. This image
        can be used to visualize car motion on the map canvas. It also plots current state image,
        step reward and next goal location in the bottom right corner of map image

        """
        
        
        # Fetch the car image rotated by current car inclination angle
        carImg = self.car.getCarImg()
        
        # Get image's dimensions
        carHeight,carWidth = carImg.shape[:2]
        
        # get car coordinates in image reference frame
        centerX,centerY = self.getCarCoordinates()
        
        # Estimate location of top left corner of car image in map canvas
        plotX = int(centerX - carWidth/2)
        plotY = int(centerY - carHeight/2)
        
        # Make a copy of original map image so that changes arre not permanently 
        # written
        
        mapImg2 = copy.deepcopy(self.mapImg)
        
        
        
        # The car could be at the border so we add padding first, plot car and then crop out the border
        
        # max width and height that 10,20 car take is 22,22 so
        # we must have border of 11. but we set border as 15 as safe margin 
        border = 15
        mapImg2 = cv2.copyMakeBorder(mapImg2, border, border, border, border, cv2.BORDER_CONSTANT,value=0 )
        
        # take care of border for car image top left corner
        plotX += border
        plotY += border
        
        # Blend car image with portion of map image at the location calculated above
        # mergeImages combines two images in proportion alpha and beta (output = alpha*img1 + beta*img2)
        blendedPart = mergeImages(carImg,1.0,mapImg2[plotY:plotY+ carHeight,plotX:plotX+carWidth ],0.0)
        
        # paste the blended part onto map image
        mapImg2[plotY:plotY+ carHeight,plotX:plotX+carWidth ]  = blendedPart
        
        # Remove the borders
        mapImg2 = mapImg2[border:border+self.mapHeight,border:border+self.mapWidth]
        
        #plot goal circles
        #Inner solid circle
        mapImg2 = cv2.circle(mapImg2,(self.goalX,self.mapHeight-1-self.goalY),8,(255,0,0),-1)
        # Outer ring representing area of radius 30 around goal
        mapImg2 = cv2.circle(mapImg2,(self.goalX,self.mapHeight-1-self.goalY),30,(0,0,255),2)
        
        
        # get Current state image
        state = self.estimateStateV2()
        
        # Make a copy of the state image
        stateImg = copy.deepcopy(state[0].squeeze())    # squeeze out extra dimesion of state image
        stateHeight,stateWidth = stateImg.shape
        
        # State image is in normalized form [0.0,1.0]. So multiply by 255.0 to bring it to [0,255]
        stateImg = np.array(stateImg*255.0,dtype = np.uint8)
        
        # Convert grayscale image to RGB because map image where we want to paste it is RGB
        stateImg = cv2.cvtColor(stateImg,cv2.COLOR_GRAY2RGB)
        
        # Paste state image onto map image
        mapImg2[500:500+stateHeight,1200:1200+stateWidth,:] = stateImg[:,:,:]
        
        # Set text color to be blue
        textColor = (255,0,0)   # in openCV order of color channels is BGR
        # Indicate 'state image'
        cv2.putText(img=mapImg2,text='State Image',org=(1170,480),fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                    fontScale=0.5,color=textColor,thickness=2)
        
        # Prepare text for showing step reward
        stepRewardMsg = 'Step Reward: %f'%self.reward
        # Write step reward onto map image
        cv2.putText(img=mapImg2,text=stepRewardMsg,org=(1115,570),fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                    fontScale=0.5,color=textColor,thickness=2)
        
        # Prepare text for showing next goal
        goalMsg = 'Next Goal: (%d,%d)'%(self.goalX,self.goalY)
        # Write next goal location onto map image
        cv2.putText(img=mapImg2,text=goalMsg,org=(1125,600),fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                    fontScale=0.5,color=textColor,thickness=2)
            
        return mapImg2
         
   
    def getCarCoordinates(self):
        # car positions are in normal coordinate form
        # we need to convert them into image coordinates first
        return (self.car.posX),(self.mapHeight-1-self.car.posY) 
        
        
