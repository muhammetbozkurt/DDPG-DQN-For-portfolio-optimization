# DDPG-DQN For Portfolio Optimization

This project applies deep learning to portfolio optimization, aiming to find optimal asset weights based on recent changes in the financial market. Portfolio optimization is a critical task in finance, involving the strategic allocation of assets to achieve expected returns while managing risk. Traditionally, financial experts use modern portfolio theory and historical asset performance to guide their decisions. However, market conditions, correlations between assets, and broader economic indicators also play a significant role. Since human emotions often influence markets, relying solely on historical data can be limiting.

Deep learning offers a way to enhance decision-making by analyzing large datasets more efficiently than humans can. As a result, many investment firms have adopted machine learning to minimize human biases. For individual investors, deep learning can similarly provide a more systematic approach to portfolio optimization by leveraging historical data.

In this project, two reinforcement learning models—Deep Q-Network (DQN) and Deep Deterministic Policy Gradient (DDPG)—are developed to determine how to distribute assets within a portfolio over time. The models learn from historical market data to optimize portfolio strategy based on the agent's experience. The assets considered include broad indices like S&P 500 and Nasdaq, currencies like USD and EUR, and commodities like gold (XAU). Given that DQN is not designed for continuous action spaces, it identifies which assets should be bought in the current state. An additional step then converts the DQN's discrete outputs into continuous values by calculating the Sharpe ratios of the selected assets.

## Results

**DDPG Results:**

![DDPG Result](https://github.com/muhammetbozkurt/DDPG-DQN-For-portfolio-optimization/blob/main/results/ddpg_result.png)

**DQN Results:**

![DQN Result](https://github.com/muhammetbozkurt/DDPG-DQN-For-portfolio-optimization/blob/main/results/dqn_result.jpg)

**Note:** GT (Ground Truth) represents the mean performance of the assets.
