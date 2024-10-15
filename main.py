from CozmoEnv import *
from QLearning import *


env = env = Cozmo_Env()
q_model_0 = Q_Learning(env)

# train model
q_model_0.learn(total_epochs = 10000, epsilon = 0.5, epsilon_decay=0.99985, epsilon_min=0.1, 
                                   lr = 0.9, lr_decay=0.99975, lr_min=0.1, discount_factor = 0.9)

# test model
q_test = q_model_0

episodes = 2
for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        obs_idx = q_test.list_to_tuple(obs.values())
        action =  np.array(q_test.get_next_action(obs_idx, 0), dtype= np.int8)

        print(action)
        
        obs, reward, done, _, info = env.step(action)
        
        score += reward
    print('Episode:{}  Score:{}    observed_orientation:{}         precise orientation:{}'.format(episode, score, obs["orientation"], env.orientation))
env.close()