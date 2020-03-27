import a3
import numpy as np

d = 13
k = 5  # number of top speakers to display, <= 0 if none
M = 8
n = 100

x = np.empty((n, d))

myTheta = a3_gmm_structured.theta('S-13A', M, d)
# print(myTheta.mu)
myTheta.precomputedforM(1)
