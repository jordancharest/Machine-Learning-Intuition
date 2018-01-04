# Upper Confidence Bound

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Ads_CTR_Optimization.csv')

# Implement Upper Confidence Bound Algorithm
import math
num_users = 10000
num_ads = 10
ads_selected = []
numbers_of_selections = [0] * num_ads
sums_of_rewards = [0] * num_ads
total_reward = 0

# Test ads on total number of users
for n in range(0, num_users):
    ad = 0
    max_upper_bound = 0
    
    # Update Confidence intervals for each ad
    for i in range(0, num_ads):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
         
        # Keep track of which ad has the highest Upper Confidence Bound
        # Display that ad next
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
     
    # Display ad, check dataset to see if user clicked or not
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

# Visualize the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()