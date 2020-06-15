# Self Driving Car

# Importing the libraries
import torch
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import os
from TD3_train import TD3
import multiprocessing
import numpy as np
np.random.seed(41)
# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, BoundedNumericProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.properties import ListProperty
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

policy = TD3(400, 2, 5.0)


last_reward = 0
scores = []

# Initializing the map
initial_upd = True
max_steps = 10000
current_step = 0
done = True
on_road = -1
on_road_count = 0
off_road_count = 0
boundary_hit_count = 0
goal_hit_count = 0
train_episode_num = 0
eval_episode_num = 0
mode="Eval"
if os.path.exists("results") == False:
    os.makedirs("results")
	
img = None
car_img = None
global_counter = 0
episode_total_reward = 0.0

# This flag full_eval_demo_mode should be enabled only for demo in Full_Eval mode.
# If you want random on road location, change the full_eval_demo_mode to False
full_eval_demo_mode = False
random_location=True



def init_upd():
    global sand
    global goal_x
    global goal_y
    global initial_upd
    global done
    global max_steps
    global current_step
    global img
    global car_img
    global global_counter
	
    sand = np.zeros((longitude,latitude))
    img = PILImage.open("./images/mask.png").convert('L')
    car_img = PILImage.open("./images/latest_triangle_car.png")
    sand = np.asarray(img)/255	
    goal_x = 575
    goal_y = 530
    initial_upd = False
    done = False
    global swap
    swap = 0
    current_step = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = BoundedNumericProperty(0.0)
    rotation = BoundedNumericProperty(0.0)
    velocity_x = BoundedNumericProperty(0)
    velocity_y = BoundedNumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.x = int(self.velocity_x + self.x)
        self.y = int(self.velocity_y + self.y)
        self.pos = Vector(self.x, self.y)
        self.rotation = rotation
        self.angle = self.angle + self.rotation
 
        
# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
	
    def serve_car(self, eventstat, resetvalue, modeupdate, stateupdate, actionupdate, nextstateupdate):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        self.car.eventstat = eventstat
        self.car.resetvalue = resetvalue
        self.car.modeupdate = modeupdate
        self.car.stateupdate = stateupdate
        self.car.actionupdate = actionupdate
        self.car.nextstateupdate = nextstateupdate


    def get_state(self, img, car_img, x, y, car_angle,longitude, latitude, on_road, hit_boundary, hit_goal, full_360_degree_rotation, distance_reduced): 
        if x - 40 <0 or y-40 < 0 or x+40 > longitude-1 or y+40 > longitude-1:
            return np.ones((80,80))
        else:				
            img_crop = img.crop((x -40, y-40, x+40, y +40))
            car_rotated = car_img.rotate(car_angle)
            car_size = (32,32)
            car_rotated = car_rotated.resize(car_size, PILImage.ANTIALIAS).convert("RGBA")
            img_crop.paste(car_rotated, (48, 48), car_rotated)
            state_value = np.asarray(img_crop)/255	
            return state_value
			
    def get_car_angle(self, car_angle):
        if car_angle > 360:
            car_angle = car_angle % 360
        elif car_angle < -360:
            car_angle = car_angle % (-360)				
        car_angle = car_angle/360
        return car_angle
		
    
	
    def select_random_on_road_location(self):
        t = np.random.randint(60, self.width-60), np.random.randint(60, self.height-60)
        while sand[t] != 0	:
            t = np.random.randint(60, self.width-60), np.random.randint(60, self.height-60)
        return t
		
    def select_demo_location(self, eval_episode_num):
        demo_on_road_postions=[(1031,496), (766,468), (881,424)]
        index = (eval_episode_num - 1) % len(demo_on_road_postions)
        return demo_on_road_postions[index]		
        

	
    def updtval(self, dt):

        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longitude
        global latitude
        global swap
        global initial_upd
        global done
        global current_step
        global max_steps
        global on_road
        global on_road_count
        global off_road_count
        global mode
        global boundary_hit_count
        global goal_hit_count
        global train_episode_num
        global eval_episode_num
        global img
        global car_img
        global global_counter
        global on_road_postions
        global random_location
        global episode_total_reward
        global stop_on_hitting_goal

        longitude = self.width
        latitude = self.height
        self.car.eventstat.wait()
        if done == True:
            reset = self.car.resetvalue.get()

            if reset == True:
                print("initial_upd is set to True")
                initial_upd = True
				
            (mode, train_episode_num, eval_episode_num) = self.car.modeupdate.get()
            print("mode: ", mode, " train_episode_num: ",  train_episode_num, " eval_episode_num:", eval_episode_num ) 
            if mode == "Train":
                max_steps = 2500

            elif mode == "Eval": 
                max_steps = 500

            else:
                max_steps = 2500
				
			
        if initial_upd:
            init_upd()
            on_road_count = 0
            off_road_count = 0
            boundary_hit_count = 0
            goal_hit_count = 0
            episode_total_reward = 0.0									
            self.car.rotation = 0.0
            self.car.angle = 0.0
			
            if mode=="Train" or  mode == "Eval":
                self.car.pos = Vector(np.random.randint(100, longitude-100), np.random.randint(100, latitude-100))                
            elif mode == "Full_Eval":
                if full_eval_demo_mode == False:
                    self.car.pos = self.select_random_on_road_location() 
                else:
                    self.car.pos = self.select_demo_location(eval_episode_num)

			
            xx = goal_x - self.car.x
            yy = goal_y - self.car.y
            if sand[int(self.car.x),int(self.car.y)] > 0:
                on_road = -1
                off_road_count += 1
            else :
                on_road = 1
                on_road_count += 1
            orientation = Vector(*self.car.velocity).angle((xx,yy))/360.
            state = self.get_state( img,car_img,self.car.x, self.car.y, self.car.angle,longitude, latitude, 0, False, False, False, False)			
            car_angle = self.get_car_angle(self.car.angle) 
            self.car.stateupdate.put((state, np.array([orientation, car_angle, 1, on_road]))) 		
   
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/360.
        hit_boundary = False
        hit_goal = False
        full_360_degree_rotation = False
        distance_reduced = False
		
		
        action_array = self.car.actionupdate.get()
        rotation = action_array[0]
        rotation = 0.6 * rotation
        velocity = action_array[1]
        new_velocity = 0.4 + 1 + velocity*0.2
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
		
        if self.car.x < 40 or self.car.x > self.width - 40 or self.car.y < 40 or self.car.y > self.height - 40:
            if mode=="Train" or  mode == "Eval":
                self.car.pos = Vector(np.random.randint(100, longitude-100), np.random.randint(100, latitude-100))                
            elif mode == "Full_Eval":
                if full_eval_demo_mode == False:
                    self.car.pos = self.select_random_on_road_location() 
                else:
                    self.car.pos = self.select_demo_location(eval_episode_num)
            last_reward = -50
            self.car.rotation = 0.0
            self.car.angle = 0.0
            boundary_hit_count += 1
            hit_boundary = True
			
        

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(new_velocity, 0).rotate(self.car.angle) 
            last_reward = -2
            on_road = 0
            off_road_count += 1
 
        else: 
            self.car.velocity = Vector(new_velocity, 0).rotate(self.car.angle)
            on_road = 1
            on_road_count += 1            
            last_reward = -0.5
            
            if distance < last_distance:
                last_reward = last_reward + 5
                distance_reduced = True
                
            else:
                last_reward = last_reward + 2
                on_road = 1
                distance_reduced = False				

        if distance < 25:
            last_reward = 100
            
            goal_hit_count += 1
            hit_goal = True

                
            if swap == 1:
                goal_x = 575
                goal_y = 530
                swap = 0
            else:
                goal_x = 610
                goal_y = 45
                swap = 1
				
            if mode == "Full_Eval" and hit_goal==True and full_eval_demo_mode==True:
                episode_total_reward += last_reward
                popup = Popup(title='Test popup', content=Label(text="Congratulations! your car has reached the destination and earned total rewards: " + str(episode_total_reward) + " during the trip"),  size=(200, 200), auto_dismiss=True)              
                popup.open()
                time.sleep(1)
                popup.dismiss()				
                done = True
				
            if mode == "Full_Eval" and hit_goal==True and full_eval_demo_mode==False:
                self.car.pos = self.select_random_on_road_location()

        last_distance = distance
		
        next_state = self.get_state(img, car_img, self.car.x, self.car.y, self.car.angle,longitude, latitude, on_road, hit_boundary, hit_goal, full_360_degree_rotation, distance_reduced)

        if self.car.angle >= 360:	
            self.car.angle = self.car.angle % 360
            last_reward += -50
           
        elif self.car.angle <= -360:	
            self.car.angle = self.car.angle % (-360)
            last_reward += -50
            
        reward = last_reward
        current_step += 1
        global_counter += 1
        if done== False:
            episode_total_reward += reward
        
        if current_step >= max_steps:
            done = True
        distance_diff = (distance - last_distance)/4
        car_angle = self.get_car_angle(self.car.angle)
        self.car.nextstateupdate.put(((next_state, np.array([orientation, car_angle, distance_diff, on_road])), reward, done, current_step))        		

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1


    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):
    def __init__(self, eventstat, resetvalue, modeupdate, stateupdate, actionupdate, nextstateupdate):
        super(CarApp, self).__init__()
        self.eventstat = eventstat
        self.resetvalue = resetvalue
        self.modeupdate = modeupdate
        self.stateupdate = stateupdate
        self.actionupdate = actionupdate
        self.nextstateupdate = nextstateupdate

    def build(self):
        parent = Game()
        parent.serve_car(self.eventstat, self.resetvalue, self.modeupdate, self.stateupdate, self.actionupdate, self.nextstateupdate)
        Clock.schedule_interval(parent.updtval, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longitude,latitude))

    def save(self, obj):
        print("saving brain...")
        policy.save("TD3_best_kivy-car_41","./pytorch_models") 

    def load(self, obj):
        print("loading last saved brain...")
        policy.load("TD3_best_kivy-car_41","./pytorch_models")


def valueInstantiator(eventstat, resetvalue, modeupdate, stateupdate, actionupdate, nextstateupdate):
    car_app_instance = CarApp(eventstat, resetvalue, modeupdate, stateupdate, actionupdate, nextstateupdate)
    car_app_instance.run()
	
class environmentresetter(object):

    def __init__(self):
        self.eventstat = multiprocessing.Event()
        self.resetvalue = multiprocessing.Queue()
        self.modeupdate = multiprocessing.Queue()
        self.stateupdate = multiprocessing.Queue()
        self.actionupdate = multiprocessing.Queue()
        self.nextstateupdate = multiprocessing.Queue()
        self.process = None
		
    def start(self):
        self.process = multiprocessing.Process(target=valueInstantiator, args=(self.eventstat, self.resetvalue, self.modeupdate, self.stateupdate, self.actionupdate, self.nextstateupdate))
        self.process.start()
        #time.sleep(10)
        time.sleep(1)
		
    def close(self):
        if self.process is not None:
            self.process.join()
			
    def reset(self, mode="Train", train_episode_num=0, eval_episode_num=0):
        self.resetvalue.put(True)
        self.modeupdate.put((mode, train_episode_num, eval_episode_num))
        self.eventstat.set()
        return self.stateupdate.get()

	
    def step(self, action):
        self.actionupdate.put(action)
        return self.nextstateupdate.get()
		
    def action_space_sample(self):
	    return np.random.uniform(-5, 5, 2)
		
    def action_space_shape(self):
        return 1
		
    def action_space_low(self):
        return -5.0
	
    def action_space_high(self):
        return 5.0

    def max_episode_steps(self):
        return 2500



if __name__ == '__main__':
    env = environmentresetter()
    env.start()
    state = env.reset()
    print("Main: Got state: ", state)
    done = False
	
    while not done:
        action = env.action_space_sample()
        obs, reward, done, _  = env.step(action)
        print("reward: ", reward, ", done: ", done, ", obs", obs)
		
    state = env.reset("Eval")
    print("Main: Got state: ", state)
    done = False
	
    while not done:
        #action = np.random.randint(3)
        action = env.action_space_sample()
        obs, reward, done, _  = env.step(action)
        print("reward: ", reward, ", done: ", done, ", obs", obs)
  
    env.close()
