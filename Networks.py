import os
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Dense

from utils import normalize

"""
https://medium.com/analytics-vidhya/tf-gradienttape-explained-for-keras-users-cc3f06276f22
If at any point, we want to use multiple variables in our calculations, all we need to do is give tape.gradient a list or tuple of those variables. When we optimize Keras models, we pass model.trainable_variables as our variable list.


This is because immediately after calling tape.gradient, the GradientTape releases all the information stored inside of it for computational purposes.
If we want to bypass this, we can set persistent=True
"""
#Environment params
TIMESTEP = 30
FEATSNUM = 6
TRAIN_TEST = "2019-05-03"
#eur xau sp nasdaq usd
INSTRUMENTNUM = 5
INSTRUMENTS = ("EURUSD=X", "GC=F", "^GSPC", "^IXIC", "USD")

#model
ACTIVATION_ACTOR = "sigmoid"
ACTIVATION_CRITIC = "sigmoid"


class CriticNetwork(keras.Model):
    def __init__(self, modelName : str = "critic", checkPointDirectory : str = "saves/ddpg") -> None:
        super(CriticNetwork, self).__init__()

        self.modelName = modelName
        self.checkpointDir = checkPointDirectory 
        self.checkpointFile = os.path.join(self.checkpointDir, 
                    self.modelName+'_ddpg.h5')

        self.inputLayer = Dense(64, activation = ACTIVATION_CRITIC, name = "input yeri")
        self.interLayer1 = Dense(64, activation = ACTIVATION_CRITIC, name = "ara layer")
        self.interLayer2 = Dense(64, activation = ACTIVATION_CRITIC, name = "ara layer")
        #self.interLayer3 = Dense(512, activation = ACTIVATION_CRITIC, name = "ara layer")
        self.q = Dense(1, name="cikti")

    
    def call(self, state, action):
        actionValue = self.inputLayer(tf.concat([state, action], axis=1))
        actionValue = self.interLayer1(actionValue)
        actionValue = self.interLayer2(actionValue)
        #actionValue = self.interLayer3(actionValue)

        qValue = self.q(actionValue)

        return qValue



class ActorNetwork(keras.Model):
    def __init__(self, nActions = 5, modelName : str = "Actor", checkPointDirectory : str = "saves/ddpg"):
        super(ActorNetwork, self).__init__()

        self.modelName = modelName
        self.checkpointDir = checkPointDirectory 
        self.checkpointFile = os.path.join(self.checkpointDir, 
                    self.modelName+'_ddpg.h5')

        self.inputLayer = Dense(64, dtype = "float32", activation = ACTIVATION_ACTOR)
        self.inputLayer_ = Dense(64, dtype = "float32", activation = ACTIVATION_ACTOR)
        self.interLayer1 = Dense(64, dtype = "float32", activation = ACTIVATION_ACTOR)
        self.interLayer2 = Dense(64, dtype = "float32", activation = ACTIVATION_ACTOR)
        #self.interLayer3 = Dense(1024, dtype = "float32", activation = ACTIVATION_ACTOR)
        #self.interLayer4 = Dense(512, dtype = "float32", activation = ACTIVATION_ACTOR)
        self.mu = Dense(nActions, dtype = "float32", activation= "softmax")

    def call(self, state):
        #state = tf.convert_to_tensor([state], dtype = tf.float32)
        prob = self.inputLayer(state)
        prob = self.inputLayer_(prob)
        prob = self.interLayer1(prob)
        prob = self.interLayer2(prob)
        #prob = self.interLayer3(prob)
        #prob = self.interLayer4(prob)

        mu = self.mu(prob)
        #total of them must be 1
        #mu = normalize(mu)
        sumOfWeights = tf.math.reduce_sum(mu)

        return mu
