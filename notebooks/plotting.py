# plotting utilities

import numpy.linalg as npl
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

def phi(pos, t1, t2):
    r1 = pos[0][t1[0]]
    r2 = pos[0][t1[1]]
    r3 = pos[0][t1[2]]

    
    p1 = pos[0][t2[0]]
    p2 = pos[0][t2[1]]
    p3 = pos[0][t2[2]]

    


    return np.degrees(angle_dot(get_normal(r1,r2,r3), get_normal(p1,p2,p3)))


def angle_dot(a, b):
    n = np.cross(b, a)
    n /= npl.norm(n)
    if n[1] < 0:
        n = -n

    return np.arctan2(np.dot(np.cross(a, b), n), np.dot(a, b))

def get_normal(a,b,c):
    p1 = c - a
    p2 = b - a
    normal = np.cross(p1,p2)
    return normal


phi_pair = (4,6,8), (6,8,14)
psi_pair = (6,8,14), (8,14,16)

def rama_plot(traj):

    sns.scatterplot(x=[phi(i, phi_pair[0], phi_pair[1]).item() for i in traj], y=[phi(j, psi_pair[0], psi_pair[1]).item() for j in traj])
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)