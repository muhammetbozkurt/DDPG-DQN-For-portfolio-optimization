import numpy as np


class ReplayBuffer:
    def __init__(self, maxSize, inputShape, nActions):
        self.memberSize = maxSize
        self.memberCounter = 0
        self.stateMemory = np.zeros((self.memberSize, *inputShape))
        self.newStateMemory = np.zeros((self.memberSize, *inputShape))
        self.actionMemory = np.zeros((self.memberSize, nActions))
        self.rewardMemory = np.zeros((self.memberSize))
        #it indicates that self.memberCounter exceeds self.memberSize at least once
        self.sizeFlag = False

        #terminal state i hafızada tutmak istemiyorum

    def storeTransition(self, state : np.array, action : np.array, reward : float, newState : np.array) -> None:

        self.stateMemory[self.memberCounter] = state
        self.actionMemory[self.memberCounter] = action
        self.rewardMemory[self.memberCounter] = reward
        self.newStateMemory[self.memberCounter] = newState

        self.memberCounter += 1
        
        if(self.memberCounter == self.memberSize):
            self.sizeFlag = True 

        self.memberCounter = self.memberCounter % self.memberSize
        

    
    def sampleBuffer(self, batchSize : int) -> np.array:
        maxIndex = self.memberSize if self.sizeFlag else self.memberCounter

        batch = np.random.choice(maxIndex, batchSize, replace = False)

        states = self.stateMemory[batch]
        newStates = self.newStateMemory[batch]
        actions = self.actionMemory[batch]
        rewards = self.rewardMemory[batch]

        return states, actions, rewards, newStates