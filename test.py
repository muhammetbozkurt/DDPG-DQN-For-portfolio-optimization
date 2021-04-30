from environmentV2 import Market
from Agent import Agent
from utils import EpisodeData

import tensorflow as tf

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


def main():
    nTurns = 10
    inputDims = TIMESTEP * FEATSNUM * (INSTRUMENTNUM - 1)

    env = Market()
    startState = env.start()

    loadCheckPoint = False

    agent = Agent([inputDims], noise=0.4 )

    episodeLogs = list ()

    #    def add(reward, totalOfGoods, weight, basket):


    for i in range(nTurns):
        state = env.reset().flatten()
        done = False
        j = 0

        episodeLog = EpisodeData()
        episodeLogs.append(episodeLog)
        while not done:
            action = agent.chooseAction(state, False)

            sumOfWeights = tf.math.reduce_sum(action)

            stateNew, reward, done = env.step(action, normalizedReward=True)
            stateNew = stateNew.flatten()
            agent.storeMemory(state, action, reward, stateNew)
            episodeLog.add(reward, env.totalOfGoods, action, env.basketCopy)

            if not loadCheckPoint:
                agent.learn()
            state = stateNew
            j += 1



if __name__ == "__main__":
    main()