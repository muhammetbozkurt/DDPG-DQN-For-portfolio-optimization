import numpy as np
import tensorflow as tf
import tensorflow.keras as keras 

from tensorflow.keras.optimizers import Adam
from ReplayMemory import ReplayBuffer
from Networks import ActorNetwork, CriticNetwork


from utils import normalize

class Agent:
    def __init__(self, inputDims, alpha = 0.01, beta = 0.02, actionMax = 1., \
        actionMin : float = 0, gamma = 0.99, nActions = 5, maxSize = 100000, tau = 0.005,\
        batchSize = 64, noise = 0.1):
        """
        alpha is learning rate for actor network
        beta is learning rate for critic network
        actionMax is max limit of action that agent can take 
        actionMin is min limit of action that agent can take 
        gama is discount foctor for update equation
        maxSize is limit for replay buffer
        tau is soft update factor
        noise is stddev valu of noise that is going to use for exploration

        beta is slightly higher than alpa and that is because in general in policy gradient type methods the policy approximation is a little bit more sensitive to
        pertubation and parameters.

        actionMax and actionMin are going to use for clipping after noise added to action for exploration
        
        default value of tau came from paper
        """

        self.gamma = gamma 
        self.tau = tau
        self._memory = ReplayBuffer(maxSize, inputDims, nActions)
        self.batchSize = batchSize
        self.nActions = nActions
        self.noise = noise
        self.maxAction = actionMax
        self.minAction = actionMin

        self.actor = ActorNetwork(nActions=nActions)
        self.critic = CriticNetwork()

        self.targetActor = ActorNetwork(nActions=nActions)
        self.targetCritic = CriticNetwork()

        self.actor.compile(optimizer= Adam(alpha))
        self.critic.compile(optimizer=Adam(beta))

        self.targetActor.compile(optimizer= Adam(alpha))
        self.targetCritic.compile(optimizer=Adam(beta))
        
        #tau is 1 because ve want hard copy
        self.updateNetworkParameters(tau = 1)


    def updateNetworkParameters(self, tau =None):
        
        tau = self.tau if tau is None else tau

        #soft update to target actor network
        networkWeights = []
        targetWeights = self.targetActor.weights
        for i, layerWeights in enumerate(self.actor.weights):
            networkWeights.append(layerWeights * tau + targetWeights[i] * (1 - tau))
        self.targetActor.set_weights(networkWeights)

        #soft update to target critic network
        networkWeights = []
        targetWeights = self.targetCritic.weights
        for i, layerWeights in enumerate(self.critic.weights):
            networkWeights.append(layerWeights * tau + targetWeights[i] * (1 - tau))
        self.targetCritic.set_weights(networkWeights) 


    def storeMemory(self, state, action , reward, newState):
        self._memory.storeTransition(state, action, reward, newState)

    def chooseAction(self, state, evaluate = False):
        state = tf.convert_to_tensor([state], dtype = tf.float32)
        actions = self.actor(state)

        sumOfWeights = tf.math.reduce_sum(actions)
        
        if not evaluate:
            actions += tf.random.normal(shape=[self.nActions], \
                mean=0.0, stddev = self.noise, dtype = tf.float32)
            actions = tf.clip_by_value(actions, self.minAction, self.maxAction)
            actions = normalize(actions)

        sumOfWeights = tf.math.reduce_sum(actions)

        return actions[0]

    def learn(self):
        if self._memory.memberCounter < self.batchSize:
            return
        states, actions, reward_, newStates = self._memory.sampleBuffer(self.batchSize)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        newStates = tf.convert_to_tensor(newStates, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward_, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        #q = reward + gamma * q_next
        #purpose of critic network is find q value of action for corresponding state
        #therefore, loss is difference of q values
        
        #with tf.GradientTape as tape:
        
        tape = tf.GradientTape()
        tape.__enter__()
        targetActions = self.targetActor(newStates)
        newCriticValue = tf.squeeze(self.targetCritic(newStates, targetActions), 1)

        criticValue = tf.squeeze(self.critic(states, actions), 1)
        
        target = rewards + self.gamma * newCriticValue
        criticLoss = keras.losses.MSE(target, criticValue)
        #tape.__exit__()

        #find gradient
        criticNetworkGradient = tape.gradient(criticLoss, self.critic.non_trainable_variables)

        #optimize weights
        self.critic.optimizer.apply_gradients(zip(
            criticNetworkGradient, self.critic.trainable_variables
        ))

        #actor try to maximize q value 
        #Hence loss is -q

        tape = tf.GradientTape()
        tape.__enter__()
        newPolicyActions = self.actor(states)
        actorLoss = -self.critic(states, newPolicyActions)
        actorLoss = tf.math.reduce_mean(actorLoss)
        #tape.__exit__()

        actorNetworkGradient = tape.gradient(actorLoss,\
            self.actor.trainable_variables)
        
        self.actor.optimizer.apply_gradients(zip(
            actorNetworkGradient, self.actor.trainable_variables
        ))

        self.updateNetworkParameters()