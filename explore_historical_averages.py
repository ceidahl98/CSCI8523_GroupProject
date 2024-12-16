import numpy as np
import matplotlib.pyplot as plt

DATA_NA_VAL = -9.96921e36

data = np.load("historical_averages.npy")

data_plot = data[364,::-1,:]
DATA_NA_VAL = data_plot.min()
data_plot[data_plot==DATA_NA_VAL] = np.nan
plt.figure()
plt.imshow(data_plot)
plt.show()