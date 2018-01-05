# Thompson Sampling

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Ads_CTR_Optimization.csv')

# Implement Thompson Sampling
import random
num_users = 10000
num_ads = 10
ads_selected = []
num_rewards = [0] * num_ads
num_punishments = [0] * num_ads
total_reward = 0
for n in range(0, num_users):
    ad = 0
    max_random = 0
    for i in range(0, num_ads):
        # Pick a random number from the distribution
        random_beta = random.betavariate(num_rewards[i] + 1, num_punishments[i] + 1)
        
        # Choose which ad we want to run next
        if random_beta > max_random:
            max_random = random_beta
            ad = i
            
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    
    if reward == 1:
        num_rewards[ad] += 1
    else:
        num_punishments[ad] += 1
    total_reward += reward

# Visualize the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()