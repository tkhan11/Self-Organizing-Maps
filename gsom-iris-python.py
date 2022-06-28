#!/usr/bin/env python
# coding: utf-8
#
### AUTHOR: TANVEER KHAN


from math import log, exp
import itertools
import math
import random
import scipy

class GSOM_Node:
    R = random.Random()
    
    def __init__(self, dim, x, y):
        # Create a weight vector of the given dimension:
        # Initialize the weight vector with random values between 0 and 1.
        self.weights=scipy.array([self.R.random() for _ in range(dim)])

        self.error = 0.0

        self.it = 0
        self.last_it = 0

        self.data = None
        self.last_changed = 0

        self.right = None
        self.left  = None
        self.up    = None
        self.down  = None

        self.x, self.y = x, y


    def adjust_weights(self, target, learn_rate):
        """ Adjust the weights of this node. """
        for w in range(0, len(target)):
            self.weights[w] += learn_rate * (target[w] - self.weights[w])


    def is_boundary(self):
        """ Check if this node is at the boundary of the map. """
        if not self.right: return True
        if not self.left:  return True
        if not self.up:    return True
        if not self.down:  return True
        return False

import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import iris data
iris = datasets.load_iris()
dataset = iris
spread_factor = 1.5
# Assign the data
data = []
for fn in dataset:
    t = dataset
    arr = scipy.array(t.data)
    data.append([fn,arr])

# Determine the dimension of the data.
dim = len(data[0][1])

# Calculate the growing threshold:
_GT = -dim * math.log(spread_factor, 2)

# Create the 4 starting Nodes.
nodes = []
n00 = GSOM_Node(dim, 10, 10)
n01 = GSOM_Node(dim, 10, 11)
n10 = GSOM_Node(dim, 11, 10)
n11 = GSOM_Node(dim, 11, 11)
nodes.extend([n00,n01,n10,n11])

# Create starting topology
n00.right = n10
n00.up    = n01
n01.right = n11
n01.down  = n00
n10.up    = n11
n10.left  = n00
n11.left  = n01
n11.down  = n10

# Set properties
it = 0       # Current iteration
max_it = len(data)
num_it = 100     # Total iterations
init_lr = 0.5     # Initial value of the learning rate
alpha = 0.2
output = file = open("gsom.csv","w")

print(len(n00.weights))


def _distance(v1, v2):
    dist = 0.0
    #print(v1)
    #print(v2)
    for v, w in zip(v1, v2):
        dist += pow(v - w,2)
    return dist


def _find_bmu(vec):
    dist=float("inf")
    winner = False
    for node in nodes:
        d = _distance(vec, node.weights)
        #print(d.all())            
        if(d < dist):
            dist = d
            winner = node

    return winner


def _find_similar_boundary(node):
    dist = float("inf")
    winner = False
    for boundary in nodes:
        if not boundary.is_boundary(): continue
        if boundary == node: continue

        d = _distance(node.weights, boundary.weights)
        if d < dist:
            dist = d
            winner = node

    return winner


def _node_add_error(node, error):

    node.error += error

    # Consider growing
    if node.error > _GT:
        if not node.is_boundary():
            node = _find_similar_boundary(node)
            if not node:
                print("GSOM: Error: No free boundary node found!")

        nodes = _grow(node)
        return True, nodes

    return False, 0


def _grow(node):
    """ Grow this GSOM. """
    # We grow this GSOM in every possible direction.
    nodes = []
    if node.left == None:
        nn = _insert(node.x - 1, node.y, node)
        nodes.append(nn)
        print("Growing left at: (" + str(node.x) + "," + str(node.y)                + ") -> (" + str(nn.x) + ", " + str(nn.y) + ")")

    if node.right == None:
        nn = _insert(node.x + 1, node.y, node)
        nodes.append(nn)
        print("Growing right at: (" + str(node.x) + "," + str(node.y)                + ") -> (" + str(nn.x) + ", " + str(nn.y) + ")")

    if node.up == None:
        nn = _insert(node.x, node.y + 1, node)
        nodes.append(nn)
        print("Growing up at: (" + str(node.x) + "," + str(node.y) +                ") -> (" + str(nn.x) + ", " + str(nn.y) + ")")

    if node.down == None:
        nn = _insert(node.x, node.y - 1, node)
        nodes.append(nn)
        print("Growing down at: (" + str(node.x) + "," + str(node.y) +                ") -> (" + str(nn.x) + ", " + str(nn.y) + ")")
    return nodes


def _insert(x, y, init_node):
    # Create new node
    new_node = GSOM_Node(dim, x, y)
    nodes.append(new_node)

    new_node.it = new_node.last_it = it

    # Create the connections to possible neighbouring nodes.
    for node in nodes:
        # Left, Right, Up, Down
        if node.x == x - 1 and node.y == y:
            new_node.left = node
            node.right = new_node
        if node.x == x + 1 and node.y == y:
            new_node.right = node
            node.left = new_node
        if node.x == x and node.y == y + 1:
            new_node.up = node
            node.down = new_node
        if node.x == x and node.y == y - 1:
            new_node.down = node
            node.up = new_node

    # Calculate new weights, look for a neighbour.
    neigh = new_node.left
    if neigh == None: neigh = new_node.right
    if neigh == None: neigh = new_node.up
    if neigh == None: neigh = new_node.down
    if neigh == None: print("_insert: No neighbour found!")

    for i in range(0, len(new_node.weights)):
        new_node.weights[i] = 2 * init_node.weights[i] - neigh.weights[i]

    return new_node


def _remove_unused_nodes():
    """ Remove all nodes from the GSOM that have not been used. """
    to_remove = []

    # Iterate over all nodes.
    for node in nodes:
        iterations_not_won = it - node.last_it
        if iterations_not_won < len(nodes) * 4.0 * (1 + it/len(data)) : continue

        if node.left:  node.left.right = None
        if node.up:    node.up.down    = None
        if node.down:  node.down.up    = None
        if node.right: node.right.left = None

        to_remove.append(node)

    # Now remove all marked nodes.
    for node in to_remove:
        print("Removing node @ " + str(node.x) + ", " + str(node.y) +               " - Current it: " + str(it) + " - Last time won: " +              str(node.last_it))
        if node.data:
            output.write(node.data + "," + str(node.x)+","+str(node.y)                + ",remove\n")
    nodes.remove(node)


## training & viz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data0, data1, data2 = [], [],[]
j = -1
for i in (dataset['data']):
    j +=1
    if j < 51:
        data0.append(i)
    elif j < 101:
        data1.append(i)
    else:
        data2.append(i)

l5 = []
print(data0[0])
for j in data0[0]:
    l5.append(j)
print(l5)
l5 = np.array(l5)
l6 = l5.transpose()
#print(np.transpose(l6))
len(point0)
#l5
#[i[0] for i in (dataset['data'])]


input = random.choice(data)[1]
learn_rate = init_lr * alpha * (1 - 3.8/len(nodes))

recalc_nodes = []
for _ in range(50):
    # best matching unit
    l0 = [i[0] for i in input]
    l1 = [i[1] for i in input]
    l2 = [i[2] for i in input]
    l3 = [i[3] for i in input]
    #print(l0)
    l0 = l3[0:50]
    BMU = _find_bmu(l0)
    BMU.last_it = it

    # Adapt the weights of the direct topological neighbours
    neighbours = []
    neighbours.append(BMU)
    if BMU.left:  neighbours.append(BMU.left)
    if BMU.right: neighbours.append(BMU.right)
    if BMU.up:    neighbours.append(BMU.up)
    if BMU.down:  neighbours.append(BMU.down)

    if BMU not in recalc_nodes: recalc_nodes.append(BMU)
    for node in neighbours:
        node.adjust_weights(l0, learn_rate)
        if node not in recalc_nodes: recalc_nodes.append(node)
    # Calculate the error.
    err = _distance(BMU.weights, l0)
    # Add the error to the node.
    growing, nodes = _node_add_error(BMU, err)
    if growing: recalc_nodes.extend(nodes)
    # Count the iteration
    it += 1
    # Re-Calc representative data elements for changed nodes.
    used_data = []
    for node in nodes:
        used_data.append(node.data)
    for node in recalc_nodes:
        dist = float("inf")
        winner = False
        winner_fn = False
        for fn,point in data:
            if fn in used_data: continue

            point0 = [i[3] for i in point]
            d = _distance(point0[0:50], node.weights)
            if(d < dist):
                dist = d
                winner = point0
                winner_fn = fn

        if node.data != winner_fn:
            node.data = winner_fn
            node.last_changed = it
        output.write(str(node.data) + "," + str(node.x) + "," + str(node.y)                + ",change\n")
        used_data.append(winner_fn)
    # Remove unused nodes.
    _remove_unused_nodes()


import pandas as pd
df = pd.read_csv('gsom.csv')
print(df.shape)
df.columns

x4 = df['10']
y4 = df['10.1']
get_ipython().run_line_magic('matplotlib', 'inline')


irissetosa_x = x1 + x2 + x3 + x4
irissetosa_y = y1 + y2 + y3 + y4 
plt.plot(irissetosa_x, irissetosa_y)



input = random.choice(data)[1]
learn_rate = init_lr * alpha * (1 - 3.8/len(nodes))

recalc_nodes = []
for _ in range(50):
    # best matching unit
    l0 = [i[0] for i in input]
    l1 = [i[1] for i in input]
    l2 = [i[2] for i in input]
    l3 = [i[3] for i in input]
    #print(l0)
    l0 = l3[50:100]
    BMU = _find_bmu(l0)
    BMU.last_it = it

    # Adapt the weights of the direct topological neighbours
    neighbours = []
    neighbours.append(BMU)
    if BMU.left:  neighbours.append(BMU.left)
    if BMU.right: neighbours.append(BMU.right)
    if BMU.up:    neighbours.append(BMU.up)
    if BMU.down:  neighbours.append(BMU.down)

    if BMU not in recalc_nodes: recalc_nodes.append(BMU)
    for node in neighbours:
        node.adjust_weights(l0, learn_rate)
        if node not in recalc_nodes: recalc_nodes.append(node)
    # Calculate the error.
    err = _distance(BMU.weights, l0)
    # Add the error to the node.
    growing, nodes = _node_add_error(BMU, err)
    if growing: recalc_nodes.extend(nodes)
    # Count the iteration
    it += 1
    # Re-Calc representative data elements for changed nodes.
    used_data = []
    for node in nodes:
        used_data.append(node.data)
    for node in recalc_nodes:
        dist = float("inf")
        winner = False
        winner_fn = False
        for fn,point in data:
            if fn in used_data: continue

            point0 = [i[3] for i in point]
            d = _distance(point0[50:100], node.weights)
            if(d < dist):
                dist = d
                winner = point0
                winner_fn = fn

        if node.data != winner_fn:
            node.data = winner_fn
            node.last_changed = it
        output.write(str(node.data) + "," + str(node.x) + "," + str(node.y)                + ",change\n")
        used_data.append(winner_fn)
    # Remove unused nodes.
    _remove_unused_nodes()


import pandas as pd
df = pd.read_csv('gsom.csv')
print(df.shape)
df.columns


x3 = df['11']
y3 = df['10']


irisverticosa_x = x0+x1+x2+x3
irisverticosa_y = y0+y1+y2+y3
plt.plot(irisverticosa_x, irisverticosa_y)

input = random.choice(data)[1]
learn_rate = init_lr * alpha * (1 - 3.8/len(nodes))

recalc_nodes = []
for _ in range(50):
    # best matching unit
    l0 = [i[0] for i in input]
    l1 = [i[1] for i in input]
    l2 = [i[2] for i in input]
    l3 = [i[3] for i in input]
    #print(l0)
    l0 = l3[100:150]
    BMU = _find_bmu(l0)
    BMU.last_it = it

    # Adapt the weights of the direct topological neighbours
    neighbours = []
    neighbours.append(BMU)
    if BMU.left:  neighbours.append(BMU.left)
    if BMU.right: neighbours.append(BMU.right)
    if BMU.up:    neighbours.append(BMU.up)
    if BMU.down:  neighbours.append(BMU.down)

    if BMU not in recalc_nodes: recalc_nodes.append(BMU)
    for node in neighbours:
        node.adjust_weights(l0, learn_rate)
        if node not in recalc_nodes: recalc_nodes.append(node)
    # Calculate the error.
    err = _distance(BMU.weights, l0)
    # Add the error to the node.
    growing, nodes = _node_add_error(BMU, err)
    if growing: recalc_nodes.extend(nodes)
    # Count the iteration
    it += 1
    # Re-Calc representative data elements for changed nodes.
    used_data = []
    for node in nodes:
        used_data.append(node.data)
    for node in recalc_nodes:
        dist = float("inf")
        winner = False
        winner_fn = False
        for fn,point in data:
            if fn in used_data: continue

            point0 = [i[3] for i in point]
            d = _distance(point0[100:150], node.weights)
            if(d < dist):
                dist = d
                winner = point0
                winner_fn = fn

        if node.data != winner_fn:
            node.data = winner_fn
            node.last_changed = it
        output.write(str(node.data) + "," + str(node.x) + "," + str(node.y)                + ",change\n")
        used_data.append(winner_fn)
    # Remove unused nodes.
    _remove_unused_nodes()
    

import pandas as pd
df = pd.read_csv('gsom.csv')
print(df.shape)
df.columns

x3 = df['10']
y3 = df['11']


irisvirginica_x = x1+x2+x3+x0
irisvirginica_y = y1+y2+y3+y0
plt.plot(irisvirginica_x,irisvirginica_y)


get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(irissetosa_x, irissetosa_y, label='irissetosa')
plt.plot(irisverticosa_x, irisverticosa_y, label='irisversicolour')
plt.plot(irisvirginica_x, irisvirginica_y, label='irisvirginica')
#plt.plot(x4,y4, label='petal width')


plt.legend(('irissetosa', 'irisversicolour', 'irisvirginica'))#, 'petal width'))
plt.legend(loc='upper center', bbox_to_anchor=(1.50, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig = plt.figure()
plt.ylim(-50, 50)
plt.xlim(-50, 50)
graph1, = plt.plot([], [], 'o')
graph2, = plt.plot([], [], '-')
graph3, = plt.plot([], [], '>')
graph4, = plt.plot([], [], '|')

def animate1(i):
    graph1.set_data(x1[:i+1], y1[:i+1])
    return graph1

def animate2(i):
    graph2.set_data(x2[:i+1], y2[:i+1])
    return graph2

def animate3(i):
    graph3.set_data(x3[:i+1], y3[:i+1])
    return graph3

def animate4(i):
    graph4.set_data(x4[:i+1], y4[:i+1])
    return graph4


ani1 = FuncAnimation(fig, animate1(200000))
ani2 = FuncAnimation(fig, animate2(200000))
ani3 = FuncAnimation(fig, animate3(200000))
ani4 = FuncAnimation(fig, animate4(200000))

plt.show()


# viz needs more iterations for better understanding, will add more. 
# currently causing bugs after a limit


## understanding output file


df.describe() # basic description of variables present

# 1 and 0 represent the x and y coordinates of the winner node


df.head(10) # first 10 records

df.head(-20)


## debugging


i = 0
for _ in range(20):
    l0 = [i[0] for i in input]
    print(l0)
    BMU = _find_bmu(l0)



l0 = [i[0] for i in input]
print(l0)
BMU = _find_bmu(l0)
print(BMU)



for fn,point in data:
    if fn in used_data: continue
    print(node.weights)


# sepal length sepal width petal length petal width
# Iris-Setosa - Iris-Versicolour - Iris-Virginica

dataset



