import pandas as pd
import numpy as np

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])
df_tr = df.T



get_ipython().magic('matplotlib notebook')

import matplotlib.pyplot as plt
import scipy.stats as st

# calculate confidence interval
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return (h, m-h, m+h)

# calculate the probability of a value in confidence interval
def probability_value_in_ci(value, mci):
    if value < mci[1]:
        return 1
    elif value > mci[2]:
        return 0
    else:
        probability = 1 - ((value - mci[1]) / (2*mci[0]))
    
    if probability < 0:
        return 0
    elif probability > 1:
        return 1
    else:
        return probability

#############################################################
# calculate the error and probability in confidence interval
#############################################################
yerr = []
prob = []
for year in [1992, 1993, 1994, 1995]:
    mci = mean_confidence_interval(df_tr[year])
    yerr.append(mci[0])
    prob.append(probability_value_in_ci(40000, mci)) # default value


import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
plt.figure()

cbar_num_format = "%.2f"

pos = np.arange(len(df_tr.columns))
years = ['1992', '1993', '1994', '1995']

# set colorbar colors
color_code = ["darkblue", "mediumblue", "royalblue", "dodgerblue", "skyblue", "white",
              "mistyrose", "lightsalmon", "darksalmon", "firebrick", "darkred"]

cmap = ListedColormap(color_code)
colors = cmap(prob)

# set colorbar bounds
bounds = [0.00, 0.09, 0.18, 0.27, 0.36, 0.45, 0.55, 0.64, 0.73, 0.82, 0.91, 1.00]
norm = BoundaryNorm(bounds,cmap.N)

plt.bar(pos, df_tr.mean(), width=1, yerr=yerr, edgecolor='black', color=colors)
plt.xticks(pos, years)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ticks=bounds,
            boundaries=bounds,
            format=cbar_num_format)

plt.show()

# use this function to get ydata
def onclick(event):
    plt.cla()
    plt.clf()
    
    yerr = []
    prob = []
    for year in [1992, 1993, 1994, 1995]:
        mci = mean_confidence_interval(df_tr[year])
        yerr.append(mci[0])
        prob.append(probability_value_in_ci(event.ydata, mci))
    
    
    cbar_num_format = "%.2f"

    pos = np.arange(len(df_tr.columns))
    years = ['1992', '1993', '1994', '1995']

    color_code = ["darkblue", "mediumblue", "royalblue", "dodgerblue", "skyblue", "white",
                  "mistyrose", "lightsalmon", "darksalmon", "firebrick", "darkred"]

    cmap = ListedColormap(color_code)
    colors = cmap(prob)

    bounds = [0.00, 0.09, 0.18, 0.27, 0.36, 0.45, 0.55, 0.64, 0.73, 0.82, 0.91, 1.00]
    norm = BoundaryNorm(bounds,cmap.N)

    plt.bar(pos, df_tr.mean(), width=1, yerr=yerr, edgecolor='black', color=colors)
    plt.xticks(pos, years)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=bounds,
                boundaries=bounds,
                format=cbar_num_format)
    
    plt.axhline(event.ydata, color = "gray")
    plt.show()



plt.gcf().canvas.mpl_connect('button_press_event', onclick)




