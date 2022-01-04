import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

df = pd.read_csv('RevisionHistory.csv')
df['Date1'] = pd.to_datetime(df['Date1']) 
df['Date2'] = pd.to_datetime(df['Date2'])

# rename columns 
df = df.rename(columns={"Card ID": "id",
                        "Date1":"date1",
                        "Date2":"date2",
                        "Interval":"interval"})

# delete unused columns
df = df.drop(["Ease factor", "Time to answer1", "Time to answer2", "Review type1", "Review type2"], axis=1)

# get a list of unique ids of the cards
card_ids = df["id"].unique()

# create logarithmic bins for the time intervals
max_time_interval = 356
time_intervals = [x**1.5 for x in range(0, int(max_time_interval**(1/1.5)) + 1)]
num_intervals = len(time_intervals)

transition_matrix = np.zeros((num_intervals, num_intervals), dtype='uint8')

for id in tqdm(card_ids):
    rows = df[df['id'] == id].reset_index(drop=True)
    time_diff = (rows['date2'] - rows['date1']).array.days
    first_interval = rows['interval'][0]
    time_diff = np.insert(time_diff, 0, first_interval)
    for i in range(len(time_diff) - 1):
        original_index = max(np.searchsorted(time_intervals, time_diff[i]) - 1, 0)
        new_index = max(np.searchsorted(time_intervals, time_diff[i+1]) - 1, 0)
        transition_matrix[new_index][original_index] += 1

norm_transition_matrix = transition_matrix.astype(float)

# the last state is capturing
norm_transition_matrix[0, -1] = 0

# norm_transition_matrix += 1e-5
norm_transition_matrix = norm_transition_matrix / norm_transition_matrix.sum(axis=0)
norm_transition_matrix[np.isnan(norm_transition_matrix)] = 0



prob_distr = np.zeros(num_intervals)
prob_distr[0] = 1
for i in range(1000):
    print(i)
    prob_distr = norm_transition_matrix.dot(prob_distr)
    if prob_distr[-1] > 0.5:
        break
    
# plot the transition matrix
fig, ax = plt.subplots(1, 2)
ax[0].imshow(norm_transition_matrix) 
forget_curve = norm_transition_matrix[0, :]
ax[1].plot(time_intervals, forget_curve)