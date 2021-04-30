import tensorflow as tf
import numpy as np


class EpisodeData:

    def __init__(self):
        self.rewards = list()
        self.totalOfGoods = list()
        self.weights = list()
        self.baskets = list()

    def add(self, reward, totalOfGood, weight, basket):
        self.rewards.append(reward)
        self.totalOfGoods.append(totalOfGood)
        self.weights.append(weight)
        self.baskets.append(basket)

    def plot(self):
        plt.plot(self.totalOfGoods)
        plt.show()



def normalize(v):
    norm = tf.math.reduce_sum(v)
    if(norm == 0):
        print(v)
        print(norm)
        result = np.array([0, 0, 0, 0, 1])
    else:
        result = v / norm

    return result