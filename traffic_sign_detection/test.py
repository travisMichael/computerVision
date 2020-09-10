
import numpy as np
import matplotlib.pyplot as plt
from ps2 import *

# n = np.sin(np.pi / 5)
#
# theta = 1.5707964
# rho = 149.0
# x = np.linspace(-5,5,100)
# y = (-np.cos(theta)/np.sin(theta))*x + rho / np.sin(theta)
# plt.plot(x, y, '-r')
#
# theta = 2.6179938
# rho = -141.0
# x = np.linspace(-5,5,100)
# y = (-np.cos(theta)/np.sin(theta))*x + rho / np.sin(theta)
# plt.plot(x, y, '-g')
#
# theta = 2.6354473
# rho = -146.0
# x = np.linspace(-5,5,100)
# y = (-np.cos(theta)/np.sin(theta))*x + rho / np.sin(theta)
# plt.plot(x, y, '-b')

# plt.title('Graph of y=2x+1')
# plt.xlabel('x', color='#1C2833')
# plt.ylabel('y', color='#1C2833')
# plt.legend(loc='upper left')
# plt.grid()
# plt.show()

my_data = np.array([
    [1, 2],
    [2, 3]
])

new_col = np.array([[0], [0]])

zeros = np.zeros((5, 2))

all_data = np.hstack((my_data, new_col))

# print(n)
print('h')