import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import math
from math import sin, cos
import pygame

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.type_aliases import GymEnv



class Cozmo_Env(Env):
    def __init__(self):
        self.limit = 10
        self.action_range = 10
        self.orientation_range = 8
        
        self.action_space = MultiDiscrete(np.array([self.action_range+1, self.action_range+1], dtype=np.int32), dtype=np.int32) # I wanted range -10 to 10 but SB3 doesn't support action values below 0
        self.observation_space = Dict({"target":        MultiDiscrete(np.array([self.limit+1, self.limit+1], dtype=np.int32), dtype=np.int32),
                                       "position":      MultiDiscrete(np.array([self.limit+1, self.limit+1], dtype=np.int32), dtype=np.int32), 
                                       "orientation":   Discrete(self.orientation_range)
                                       })
        
        self.screen_size = np.array([960,960], dtype =np.int32) # default surface size
        self.surf_size = (self.screen_size * 0.9).astype(np.int32) # This is to fit the position limits around 90% of the display surface
        self.surf_position_ratio = self.surf_size  / (self.limit+1)   
        self.screen = None
        self.surf = None
        self.surf_observations = None
        self.target_size = 20        
        self.wheel_distance = 1
        self.reset()


    def reset(self, *, seed: int | None = None, options: dict[str, any] | None = None):
        super().reset(seed=seed)

        self.state = self.observation_space.sample()
        target = round(self.limit - self.limit/10)
        self.target = np.array([target,target], dtype=np.int32)
        self.position = np.array([self.limit/2, self.limit/2], dtype=np.int32)
        self.orientation = 0
        self.speed = np.array([self.action_range/2, self.action_range/2], dtype=np.int32)
        self.state.update({'position': self.position, 'orientation': self.orientation, 'target': self.target})

        self.count_down = 60
        self.transformation_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])

        #if self.screen is not None: 
        self.position_path = ()
        self.target_rect = pygame.Rect(((self.state["target"][0] * self.surf_position_ratio[0] + self.surf_position_ratio[0]/2 - self.target_size/2), 
                                        (self.state["target"][1] * self.surf_position_ratio[1] + self.surf_position_ratio[1]/2 - self.target_size/2), 
                                        self.target_size, self.target_size)) # needs to be changed to accommodate window resizing, possibly put in render()

        return self.state, {}


    
    def step(self, action):
        self.speed = np.array((action - (self.action_range/2)), dtype=np.int32)  # to account for the action space range being between 0 to 100 instead of -50 to 50
        self.count_down -= 1
        

        distance = self.speed * 0.03  # distance = speed * time
        
        theta = (distance[1] - distance[0]) / self.wheel_distance

        # todo: what if left wheel is the exact negative of right wheel? look at lecture slides and implement code for this turning in place scenario 
        if abs(theta) < 0.01:
            transform = np.array([[   1,    0,  distance[0]], 
                                  [   0,    1,     0       ], 
                                  [   0,    0,     1       ]])
        else:
            radius = (distance[1] + distance[0]) / (2*theta)
            transform = np.array([[cos(theta),  -sin(theta),        radius * sin(theta) ], 
                                  [sin(theta),   cos(theta),  -radius * (cos(theta) -1) ], 
                                  [         0,            0,              1             ]])

        
        old_x, old_y = self.position
        self.transformation_matrix = self.transformation_matrix @ transform

        y, x = self.transformation_matrix[0:2,2] + self.limit/2 # x and y coord from transformation matrix + half the coord limit (starting coord)
        self.position = np.array([x,y], dtype=np.float16)
        
        # transforming theta from radians to degrees then adding to orientation, with modulus 360 at the end to keep values from spinning out of range
        self.orientation = (self.orientation + (theta/math.pi)*180) % 360

        old_distance = math.sqrt((self.target[0]-old_x)**2 + (self.target[1]-old_y)**2 )
        distance = math.sqrt((self.target[0]-x)**2 + (self.target[1]-y)**2)
        
        reward = 0        
                                        # REWARD BASED ON DISTANCE TO TARGET
        if (old_distance - distance) > 0:
            reward = ((old_distance - distance) * 50)**2
        else:    
            reward =  -( ((old_distance - distance) * 80)**2 )

        #print("distance reward:", reward)
        
                                        # REWARD FOR ARRIVING ON TARGET
        if (distance <= 1 and (self.speed == [0,0]).all()):
            reward += 100
        #print(distance)
        #print(self.limit*0.05)
                
                                        # REWARD FOR EXCEEDING BOUNDARY LIMITS
        if( x >= self.limit or y >= self.limit):
            reward -= 500

        
        self.state['position'] = np.clip(np.round([x,y]).astype(np.int32), 0, self.limit)
        self.state['orientation'] = round( self.orientation / (360/self.orientation_range) ) % self.orientation_range
    
        if self.count_down <= 0:
            done = True
        else:
            done = False
            
        info = {}

        return self.state, reward, done, False, info

    
    
    def render(self):
        import pygame
        
        if self.screen is None:            
            pygame.init()
            self.backround = (137,157,172)
            
            self.screen = pygame.display.set_mode((self.screen_size[0]*2, self.screen_size[1]), pygame.RESIZABLE)
            pygame.display.set_caption("cozmo training sim")
            
            self.cozmo_sprite = pygame.image.load("cozmo_sprite.png").convert_alpha()
            #self.target_rect = pygame.Rect((  (self.state["target"][0] * self.surf_position_ratio[0] + self.surf_position_ratio[0]/2), 
            #                                  (self.state["target"][1] * self.surf_position_ratio[1] + self.surf_position_ratio[1]/2), 
            #                                  20, 20))

        
        pygame.time.delay(100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()


        
        self.screen_scale_ratio = np.array(self.screen.get_size()) * [0.5, 1] / self.screen_size

        self.surf = pygame.Surface(self.surf_size)
        self.surf.fill(self.backround)

        pygame.draw.rect(self.surf, (255,0,0), self.target_rect)

        # Draw grid of the observable positions
        positions_pxl = (self.surf_position_ratio[0]).astype(np.int32)
        for x in range( 0, self.surf_size[0], positions_pxl ):
            pygame.draw.line(self.surf, (255,0,0), (1,x), (self.surf_size[0], x), 2)
            pygame.draw.line(self.surf, (255,0,0), (x,1), (x, self.surf_size[0]), 2)


        
        #----------------------------------------Display environment based on the observations-------------------------------------
        self.surf = pygame.Surface(self.surf_size)
        self.surf.fill(self.backround)
        
        self.cozmo = pygame.transform.rotate(self.cozmo_sprite,  self.state["orientation"] * (360/self.orientation_range) )
        self.cozmo = pygame.transform.flip(self.cozmo, True, True)
        x, y = np.round(self.state['position'] * self.surf_position_ratio + self.surf_position_ratio / 2)

        self.cozmo_rect = self.cozmo.get_rect(center=(x, y) ) # this is done to rotate the cozmo srpite image around it's centre
        self.surf.blit(self.cozmo, self.cozmo_rect)

        pygame.draw.rect(self.surf, (255,0,0), self.target_rect)

        # Draw grid of the observable positions
        positions_pxl = (self.surf_position_ratio[0]).astype(np.int32)
        for x in range( 0, self.surf_size[0], positions_pxl ):
            pygame.draw.line(self.surf, (255,0,0), (1,x), (self.surf_size[0], x), 2)
            pygame.draw.line(self.surf, (255,0,0), (x,1), (x, self.surf_size[0]), 2)
        
        self.surf = pygame.transform.scale(self.surf, self.surf.get_size() * self.screen_scale_ratio)
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, self.screen_size * 0.05)

        
        #-----------------------------------------------Display environment with precise positions------------------------------------------------
        self.surf = pygame.Surface(self.surf_size)
        self.surf.fill(self.backround)
        
        self.cozmo = pygame.transform.rotate(self.cozmo_sprite,  self.orientation )
        self.cozmo = pygame.transform.flip(self.cozmo, True, True)
        x, y = np.round((self.position * self.surf_position_ratio) + self.surf_position_ratio / 2) 

        
        # draw path that the agent has traveled
        self.position_path += ((x,y),)
        for i in range(len(self.position_path)):
            pygame.draw.rect(self.surf, (255,255,255), pygame.Rect(self.position_path[i][0], self.position_path[i][1],4,4) )

        self.cozmo_rect = self.cozmo.get_rect(center=(x, y) ) # this is done to rotate the cozmo srpite image around it's centre
        self.surf.blit(self.cozmo, self.cozmo_rect)

        pygame.draw.rect(self.surf, (255,0,0), self.target_rect)

        # Draw grid of the observable positions
        positions_pxl = (self.surf_position_ratio[0]).astype(np.int32)
        for x in range( 0, self.surf_size[0], positions_pxl ):
            pygame.draw.line(self.surf, (255,0,0), (1,x), (self.surf_size[0], x), 2)
            pygame.draw.line(self.surf, (255,0,0), (x,1), (x, self.surf_size[0]), 2)
        
        self.surf = pygame.transform.scale(self.surf, self.surf_size * self.screen_scale_ratio)
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, self.screen_size * [1.05, 0.05])

        
        pygame.display.update()


      

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

