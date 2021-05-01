# DDPG-DQN-For-portfolio-optimization

Portfolio Optimization with Deep Learning project is a project which try to find optimal weights in every instrument in portfolio according to current last changes in the financial market. One of the critical jobs in finance is the movement and returns of the instruments in the portfolio within the expected time frame. The risk taken should be taken into account when making the weight distribution in the portfolio. Many finance experts evaluate modern portfolio theory and the movement of the instrument in the past years while distributing portfolios. Not only the movement of the instrument itself, but also the market conditions, the state of other instruments, and many indices related to the working and living conditions of people around the world. Although the distribution is created in the light of financial data, it is the people who make up the market and it is almost impossible for people to act without using their emotions.

As in many fields, deep learning can be used in finance to increase efficiency and to learn with the help of historical data. It is much more advantageous in terms of time and margin of error for a computer to analyze the big data that a person will examine and analyze.

At this point, many investment companies use machine learning to remove the human factor. However, there is no changing attitude for the individual investor. It is usual to have an efficient portfolio optimization with historical data thanks to deep learning. For that, two models were developed that analyzes how investment instruments should be distributed over time within a portfolio by implementing reinforcement learning techniques.

2 different model which based on DQN and DDPG reinforcement learning techniques used in this work. An agent learn from market historical data and optimize portfolio distribution strategy based on agent experience. Instruments used are more generic indices like S\&P and Nasdaq, currencies like USD and EUR, commodity which is XAU(Gold Price per ounce). Because DQN is not suitable for continous action space, unlike DDPG, it just try to find which of these instuments should be bought relative to current state. After intruments determined by DQN agent, one more step added to convert DQN agent's discrite outputs to continous ones. This added step simply calculates sharpe values of insturuments in DQN's output.

# Results

___DDPG result:___

![Figure 1](https://github.com/muhammetbozkurt/DDPG-DQN-For-portfolio-optimization/blob/main/results/ddpg_result.png)

___DQN result:___

![Figure 2](https://github.com/muhammetbozkurt/DDPG-DQN-For-portfolio-optimization/blob/main/results/dqn_result.jpg)

__Note:__ GT (Ground Truth) is mean of instruments
