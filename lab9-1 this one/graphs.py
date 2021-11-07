import numpy as np
from matplotlib import pyplot as plt


arr = np.loadtxt("./reinforce_standard_scores.txt")

for i in range(arr.shape[0]):
    length = np.arange(start = 0 , stop = arr.shape[1])
    plt.plot(length, arr[i])
    plt.show()