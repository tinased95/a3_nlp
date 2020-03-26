import a3_gmm
import numpy as np

d = 5
m = 8

x = np.empty(d)

myTheta = a3_gmm.theta('S-13A', m, d)
# print(myTheta.mu)
a3_gmm.log_b_m_x(m, x, myTheta)
