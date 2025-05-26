# WorldQuant-Capstone-MScFe690-Stress-Period
Capstone project for MScFE from WorldQuant University (Group 9184). Analysis carried out on a stress period (Great Financial Crisis)
# DynaCAAST-Framework-for-RL-based-trading-agents
This is a project carried out for capstone required for completing MScFE from WorldQuant University (Group 9184). We repeat the entire exercise carried out earlier for 2021-2025 timeframe for financial stress period as well to confirm the efficacy of our models and DynaCAAST framework. For this purpose, we choose the Great Financial Crisis and use the period  2006-01-01 to 2009-12-31.

## **Concepts**

Reinforcement learning (RL) is a part of machine learning applied to a sequential decision-making process. It involves an agent that seeks to achieve its objective, also referred to as reward, by gathering information from its environment. The interaction of an agent with its environment is accomplished through state, which is the representation of the environment for the agent. Through its interaction with the environment as represented by the state presented to the agent, the agent learns a policy that maximizes its cumulative reward. Policy is an action that an agent takes when presented with a certain state. The key idea is that in a sequence of steps the agent receives a state and takes an action and receives a reward and a new state representation. The agent learns its policy that will maximize the cumulative reward over muti-steps. The core elements of an RL algorithm are the following:

1.   **Agent**. The entity that interacts with the environment and acts based on the state it receives from the environment. For example an agent can be a trading algorithm that considers the prices provided an exchange and determines its action which can be buy or sell or allocate.

2.   **Environment**. The universe in which the agent exists. Environment provides its current situation to the agent through state, provides rewards based on the action the agent takes and indicates the change in situation through a new state. In the case of stock prices, environment can be the stock exchange that provides the closing prices of stocks as the state.

3.  **State**: The representation of the environment as provided to the agent. In our project, we experimented with multiple versions of states - technical indicators, lagged returns, LSTM model forecast and transformer forecast.

4. **Action**. This is what an agent takes when provided with a state. In this assignment, the weights allocated to a portfolio of stocks is the action that an agent takes.

5. **Reward**. This is the feedback that the environment provides to an agent based on the action an agent takes. We considered the portfolio value as the reward for the agent.

6. **Policy**. Policy refers to the action that an agent takes based on the state that the environment offers to the agent. In our case, policy would mean what allocation of weights to be done based on the state received.

<BR>
Markov Decision Process (MDP) is the mathematical foundation for RL algorithms and provides these algorithms with an executable framework. MDPs are based on Markovian property that the future state is dependent on current state and the action taken by the agent. This assumption greatly simplifies the solving of the RL problems. As we see in this capstone, this assumption does not dilute the effectiveness of the RL agents.
<BR>
In this capstone, we use policy-based algorithms that are well suited to the problem we are addressing. PPO, A2C, DDPG, TD3 and SAC are the policy-based algorithms that we would use in this study. We use offline RL algorithms as we endeavour to build as autonomous trading agent that can be deployed on live data. We briefly describe these algorithms below:

1.   **Proximal Policy Optimization (PPO).** PPO is an on-policy algorithm that uses data resulting from the current policy. Its  key aspect is that it facilitates stable learning by not allowing large updates to policy. It is known for its simplicity, stability and efficiency.

2.  **Advantage Actor-Critic (A2C).** A2C leverages the advantages of both policy-based and value-based algorithms. It has two parts – an actor and a critic. An actor takes the action based on the state presented to it. A critic indicates the value of a state or state-action pair that helps the actor take the appropriate action. However, as an on-policy method, A2C suffers from the curse of sample inefficiency.

3. **Deep Deterministic Policy Gradient (DDPG).** DDPG operates for processes that have continuous actions. It is model-free, and unlike above, it is an off-policy algorithm that uses the actor-critic logic described above.  Also, unlike the above algorithms that have stochastic policies, DDPG has a deterministic policy, which means that for a particular state, it has a particular action. Also, as it is based on neural networks to define policy and value functions, it is sensitive to choice of hyperparameters.

4. **Twin Delayed Deep Deterministic Policy Gradient (TD3).** TD3 is an advancement over DDPG addressing its value over-estimation and instability. It overcomes over-estimation limitations in DDPG by employing two critics and using the lower of them as 'q-value'. The instability issue is overcome by delaying policy updates vis-à-vis ‘critic’ updates and introducing noise in the actions in ‘critic’ updates. Like DDPG, it uses target networks and experience replay for stability.

5. **Soft Actor-Critic (SAC)**. SAC is like actor-critic algorithms operating in continuous actions space but additionally incorporates entropy within the reward structure. This brings about greater stability by encouraging greater exploration. Unlike DDPG and TD3, SAC employs stochastic policy as opposed to deterministic policies. The additional feature, however, introduces greater computing complexity.

## **Data, Transformation and Design**

### *Description of Dataset*
We select a time-period from January 1, 2021, to April 30, 2025, for our study that includes training, testing and validation periods. The training data, testing data and validation data are split in the ration of 75%:20%:5%. With this split, the end dates for training data and testing data are 2024-03-28 and 2025-02-05 respectively. This choice of time-period is reflective of medium-term trading agent that needs to be trained on a certain period of time and tested over a relatively medium timeframe (~1 year). Based on the perusal of published literature, about three to four years of training data should allow our agents to decipher sustained and stable elements in data to trade for the following year without recalibration. This is also the essence of offline trading agents that are effective as autonomous trading agents.
<BR>
We used the Yahoo Finance website as our main data source as this platform provides the required data in easily extractable form (for programming in Python). Through this website, we gather datasets for each of the chosen stocks that contain daily open price, high price, low price and close price and volumes for all these stocks. We also gather daily index values for NIFTY 50 for the chosen time horizon.
<BR>
The portfolio constituents for this analysis are 26 companies that form part of the NIFTY 50 index. We also consider CASH as one of our 'ticker' to allow our RL agents the flexibility of responding to systemic events. Our objective is for the RL agents to output weights of these companies within the portfolio such that at the end of the investment horizon, the value of the portfolio will be  maximized. The rebalancing will be performed daily, and trading of these stocks will be towards ensuring the weights forecast by RL agents. To have a realistic market scenario, we consider a transaction of 0.3% of the transacted amount.
### *Description of Algorithms*
The objective of our study is to gauge the performance of traditional portfolio management techniques before drawing comparisons between them and the RL algorithms. These traditional portfolio management techniques constitute our baseline or benchmark algorithms. The algorithms chosen for this study include Markowitz means-variance theory, De Prado denoising, equal-weighting and Kelly’s Criterion. Based on the outcome of the analysis, we choose the best performing RL and traditional algorithm to be compared against NIFTY buy-and-hold strategy. We also develop a **DynaCAAST framework** that mitigates the inherent variability observed in offline RL agents.
<BR>
<BR>
The RL algorithms used for this study include A2C, PPO, TD3, SAC and DDPG and were developed utilizing the FinRL library created by Liu et al. and the Stable-Baselines3 package. For these RL agents, we have defined four different state representations. First are the technical indicators that are typically used by stock technicians to predict stock price movements. These include Average True Range, Bollinger Bands, On-balance volume, Moving Average Convergence Divergence, Average Directional Movement Index, Simple Moving Average, Exponential Moving Average, Commodity Channel Index, and  Relative Strength Index. We also use lagged returns as the state. For this purpose, we use the lagged returns for 1 day, 2 days, 5 days, 21 days and 30 days. Next, we train an LSTM model that predicts the return for the following day. Its output will represent ‘states’ for our RL agents. Finally, we also use the forecasts provided by transformer model as the state of our agents.
<BR>
<BR>
We use Hidden Mark Model to identify and understand regime changes using NIFTY 50 index as the proxy for our portfolio. We also use correlation analysis to understand dependencies and relationships within the stocks in our portfolio. This will also help explain the portfolio results we observe.
<BR>
### *Data transformations*
Based on the raw data that has been extracted, certain transformations are carried out for the purposes of calibrating and testing these algorithms. These transformations include identifying and addressing missing data, creating lagged returns, covariances, and creating technical indicators.  We also define, train and run an LSTM model and transformer model and use their outputs as states of our RL agents. After all transformations have been performed on the dataset, this dataset is split into training, testing and validation sets in the proportion of 75%, 20% and 5% to train and evaluate our algorithms. As our dataset is a financial timeseries data, we have split the data into a training time horizon, testing time horizon and validation time horizon.

### *RL setup*
The RL environment is defined as the stock environment that take an action and provides back reward in the nature of portfolio value based on the weights provided (which is action). We use multiple state representations that include technical indicators, lagged returns, LSTM model and transformer model outputs. Once an action is taken by the agent that is the portfolio weights, the environment provides the agent with a new state. The reward function is defined as the cumulative gain on the portfolio at the end of investment horizon. We have chosen this reward as we expect it to encapsulate all aspects that ultimately result in maximum portfolio gain. This generally the objective of most investors.
<BR>
### *Performance measures*
We use the following measure to gauge the performance of traditional algorithms, RL agents and DynaCAAST framework. These measures are commonly used in financial analysis.
*   Cumulative returns
*   Maximum drawdown
*   Sharpe’s ratio
<BR>
For analysis of best performing algorithm and RL agent with the buy and hold strategy, we expand the above measures further and include metrics like Sortino’s ratio for evaluating results from DynaCAAST framework.

# **Code explanation**

The code written **(Main notebook.ipynb)** for this capstone has been compartmentalized as multiple section with each section capturing a particular functionality or activity. This is to enable ease of understanding the entire code.

## *Section 1: Python library installations and imports*

### Installations

For this project, we need several Python libraries that are not a part of Google Colab, where we performed our experiments. These include FinRL, stable-baselines and gym libraries for developing reinforcement learning (RL) agents and environment. yfinance the Yahoo Finance source for our data. ta for allowing us to create technical indicators as our 'states' for RL agent. pyfolio for enabling graphical dispaly of performance and comparison metrics. hmmlearn for regime change detection within our data using Hidden Markov Model. tensorflow for training our LSTM and transformer models.

### Imports

Imports follow the libraries that we have installed above. stable baseline provides us with various RL agents based on a particular algorithm (DDPG, PPO, TD3, SAC and A2C). Other imports are primarily the usual imports required for a financial project like pandas, numpy, etc. We also use teansorflow and keras for LSTM and transformer models.

## *Section 2: Defining functions and classes*
In this section, we define functions and classes that are necessary for this capstone project.

### Functions

In this section, function is defined to create technical indicators as our states. This function takes close prices, open price, high and low price as inputs and outputs the technical indicators of the chosen stock tickers. The function utilizes ta library to achieve this functionality. Most of the remaining functions are defined primarily to achieve denoising technique as part of our benchmark techniques. These functions are based on the previous courses completed at WorldQuant University. Finally, we define one function to facilitate analyzing performance statistics like maximum drawdown.

### Classes
In this section, classes have been defined for the purpose of embedding functionality to our reinforcement learning (RL) agent and environment.

## *Section 3: Defining 'States' for our Reinforcement Learning Agent*

In this section, we define states for our reinforcement learning (RL) agent. States are what the RL environment provides to the RL agent in order to make a decision or take an action. For this capstone, we explore four different state repesentations to explore the effectiveness of state definition on RL agent's performance. We define the four variants of the states that we use for running for RL algorithms. These states include technical indicators, lagged returns, forecasts from LSTM and transformer models.

## *Section 4: Defining model parameters*

We now define our model parameters for the various RL agents that we propose to use. We also define dictionaries to hold environments and observations for running our DynaCAAST framework.

## *Section 5: Creating training, testing and validation data*

In this section, we extract and transform our data to be suitable for running our models. In this section, we also define start date, end date, and stock tickers. The feature engineered data for all defined states for training, testing and validation of various states is also created in this section.
<br><BR>
Next, we extract and transform our data. We use Yahoo Finance to source our data. Notice that we also introduce 'CASH' as one of the tickers. We do so to allow our agents to move investments from stock to cash and vice versa when there is a bearish or bullish phase when all stocks are expected to behave a symmetric and coordinated manner.
<br><br>
We create data for our state representing technical indicators. This is followed by creating a lag returns data for evaluating states that represent lagged returns. Next, we defining start and end dates for our training, testing and validation data.
<br><br>
We now create data for state representing predictions from our LSTM model now. We do this now because we have to training LSTM model based on training data and it can only be done once training, testing and validation dates are determined. We first create training data to train a LSTM model and then use this model to spew out our 'state' representation. We check performance of our LSTM output to validate its role as a state for our RL agents. We then create the data for LSTM states. Next, we create data for states representing predictions from our Transformer model. For this we first create training data to train a Transformer model and then use this model to spew out our 'state' representation. We carry out this exercise in a separate notebook as the training requires GPU as opposed to CPU used here. We import here the forecast data that we created from our transformer model in another notebook. Finally, we create data for training, testing and validation for all 'states' of RL agents.
<br><br>
## *Section 6: Analysing correlations and regime changes in time series data of closing prices*
We analyse correlations and regime changes in our time series data as this provided explanatory background and context for the observed portfolio performances.

## *Section 7: Training and testing of traditional and RL agents*

In this section, we train and test all our approaches/agents with the training and testing data that we had created earlier. This section occupies largest code base as not only all traditional and RL agents are trained and tested, but the performance of RL agents sharing the same states are analysed as part of this section. We reserve the validation dataset to be used later to evaluate the performance of DynaCAAST framework against Nifty 50 index. In this section all agents, traditional and RL, are first trained on the training data that we have created before. Then these agents are evaluated against the training and test data. Their performances on both training and testing are graphically presented and all performance metrics are calculated for performance analysis later.

## *Section 8: Performance Analysis of Traditional Approaches on testing dataset*

In this section, we perform an analysis of traditional algorithms to understand the causes of this performance.

## *Section 9: Performance Analysis of all traditional and RL agents*

In this section, we analyse the performance of all trading agents - traditional and RL based in this section. We first evaluate against the training data (in the sample) and then on testing data (out-of-sample).

## *Section 10: The DynaCAAST Framework for RL based trading agents (Testing Data)*

In this section, we introduce the DynaCAAST framework for RL based trading agents. The framework is designed to counter the inherent variability in performance of various RL agents on a particular time series data owing to nature of training of neural networks, role hyperparameter tuning, impact of how states have been design etc. This variability does not allow a user to be confident of the performance of a particular RL agent. To counter this issue, this framework is designed. In this section, we notice that DynaCAAST framework manages the variability in performance while also ensure superior performance. In this section, review the performance of DynaCAAST Framework on testing dataset. First we clear any old elements in dictionaries for holding environments and observations.Then we aggregate the performance of all approaches and the DynaCAAST framework we draw inference on overall experiments on testing data.

## *Section 11: The DynaCAAST Framework for RL based trading agents (vs. Nifty Index)*

In this section, we benchmark our DynaCAAST algorithm against Nifty 50 index. That is we compare how our DynaCAAST framework fared versus an investor just holding the Nifty 50 index. We create multiple environments for training datasets for our RL agents. We need this as we want to evaluate our DynaCAAST framework against Nifty 50 index for the entire duration of the time period. Once we DynaCAAST results on training and testing sets and Nifty 50 returns, we use pyfolio module to compare and contrast these performances.

## *Section 12: The DynaCAAST Framework versus a buy and hold strategy (Validation Data*

In this section, utilizing our DynaCAAST framework, we run it again on the validation data to confirm its effectiveness. We then evaluate the performance of DynaCAAST Framework with buy and hold strategy on the validation data.

# **Transformer Module**
<BR>
In order to create state space using forecasts from transformer model, we create a separate notebook (TransformerModule.ipynb). This is to facilitate running on GPU as opposed to running on CPU for the above main notebook. The notebook organization is fairly self-evident. In Section 1, we import our libraries. In Section 2, we define few functions needed for our transformer model. Section 3 defines, the start and end dates and the tickers. In section 4, we train our model. In Section 5, we evaluate the model just as we did for LSTM model. In Section 6, we create our features file in csv format for use in the main notebook.
