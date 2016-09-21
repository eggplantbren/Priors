import numpy as np
import matplotlib.pyplot as plt

# The "two balls in a bag" example
# Hypotheses BB, BW
# Data B, W

# Conditional prior
P_D_given_H = np.array([[0.0, 0.5], [1.0, 0.5]])

# Compute mutual information as a function
# of the one free unknown, P(BB).
def mutual_information(P_BB):
    P_BW = 1.0 - P_BB
    P_H = np.array([[P_BB, P_BW], [P_BB, P_BW]])
    P_joint = P_H*P_D_given_H
    P_D = np.array([P_joint.sum(axis=1), P_joint.sum(axis=1)]).T

    return np.sum(P_joint*np.log(1E-300 + P_joint/(P_D*P_H + 1E-300)))


P_BB = np.linspace(0.0, 1.0, 1001)
I = np.empty(len(P_BB))

for i in range(0, len(P_BB)):
    I[i] = mutual_information(P_BB[i])

plt.plot(P_BB, I, "k")
plt.show()


# Reference prior is (0.6, 0.4)
# Entropic prior is  (1/3, 2/3)
# Jeffreys Prior is undefined (discrete space) <- screw that
# 
