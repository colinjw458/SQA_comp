# Real-time Machine Learning Stock Portfolio Selection

This paper presents a real-time portfolio optimization model using a Deep Deterministic Policy Gradient (DDPG)-based agent combined with the CSP library to handle real-time, asynchronous market data. Traditional portfolio optimization techniques often rely on static, periodic rebalancing, which may not adequately respond to the rapid fluctuations of the financial markets. Our approach introduces a dynamic, real-time machine-learning mechanism that continuously adapts to market change. While the current implementation uses historical data for simulation purposes, it can be easily adapted to process live market feeds with minimal code changes. Currently, each agent adjusts their positions dynamically based on second-interval price updates. The results show the effectiveness of the DDPG model in optimizing portfolio allocation under real-time constraints, with key findings detailed below.


## Key Features

### 1. Real-time Data Processing with CSP

The system is designed to handle real-time, asynchronous market data using CSP. While the current implementation uses historical data for simulation purposes, it can be easily adapted to process live market feeds with minimal code changes. This flexibility allows for seamless transition between backtesting and live trading environments.

#### Benefits of CSP

Multiple Real-Time Data Steams: The project uses multiple portfolios managed by different agents. CSP allows each of these portfolios to be updated (nearly simultaneously) as data is received, greatly improving efficiency, and allowing each portfolio to react to market changes in real-time.

Asynchronous Data Handling: The system deals with price data for multiple stocks (NVDA, AAPL, MSFT) that may arrive at different times. CSP's model is well-suited for managing these asynchronous updates, ensuring that each portfolio can react to new information as soon as it becomes available.

Scalability: If you decide to add more stocks or create additional portfolios, CSP makes it straightforward to scale up the system. New data streams or portfolio processes can be added without major restructuring of the existing code.

Efficient Resource Use: By leveraging CSP, the system can make better use of available computing resources, potentially speeding up the simulation and allowing for more complex strategies or larger datasets.

Clear Data Flow: The CSP graph in portfolio_manager_graph() clearly shows how data flows through the system, from price inputs to portfolio value outputs. This clarity makes the system easier to understand and modify.

### 2. Q-Learning Reinforcement Learning

We employed Q-Learning for portfolio optimization to allow an agent to learn optimal actions through iterative interactions with the environment, aiming to maximize cumulative rewards without prior knowledge of price movements. The price movements in an equity market are highly stochastic which could be difficult to model on a high level. Q-Learning provides a model-free learning method where the agent adjusts directly to market dynamics without needing a predefined model.

We defined a Q-Learning environment simulating a trading portfolio, implemented in the PortfolioEnv class. The state of the environment consists of the cash balance and the allocation of the three stocks. Each training episode begins with randomly assigned holdings of stocks and cash. The state captures both current market conditions and portfolio status, including stock prices and portfolio holdings.

The agent operates in a continuous action space, where it can take proportional actions to Buy, Sell, or Hold shares of each stock, meaning the agent doesn't simply take discrete actions but can decide how much of each stock to buy or sell, which better reflects the realities of trading. After each action, the environment transitions to a new state, and the agent receives a reward based on the percentage change in total portfolio value.

To handle this continuous action space, we implemented a Deep Deterministic Policy Gradient (DDPG) algorithme[[1]](Continuous control with deep reinforcement learning). This approach allows the agent to learn continuous actions by using two neural networks: an actor-network, which predicts the action (i.e., the proportion of the portfolio to buy, sell, or hold), and a critic network, which evaluates the expected rewards of those actions. These networks work together to optimize the agent’s decision-making process over time.

Additionally, the OUActionNoise class generates smooth, temporally correlated noise, which facilitates the exploration of the continuous action space. This noise allows the agent to explore actions in a realistic, gradual manner, avoiding abrupt or unrealistic changes in portfolio allocation during training.

Finally, we fine-tuned training parameters, such as learning rate and discount factors, to optimize the learning process and improve the agent’s performance.

### 3. Model Findings

In the sample outputs above, we have trained the model on the top three performing stocks in the S&P 500 index - NVDA, MSFT, and AAPL. Each training session began with three portfolios, initialized with randomized stock holdings and a total portfolio value of $10,000. To simulate real-time data, we used second-interval historical stock prices from randomly selected trading days to mimic the dynamic nature of live trading environments and allow the agent to learn and adapt over time.

The DDPG algorithm was used due to its suitability for continuous action spaces, enabling the agent to take more granular actions (i.e., buy/sell fractional shares) rather than discrete ones. The agent’s performance was evaluated against the individual stocks.

The training results on three portfolios for a single trading day showed that DDPG-based portfolios consistently outperform the benchmark portfolio across key financial metrics:

Final Portfolio Value: The agent was able to achieve higher terminal values for the portfolios compared to the static equal-weight approach, indicating its ability to capitalize on favorable stock movements and optimize allocations dynamically.

Sharpe Ratio: The trained DDPG agent also exhibited a higher Sharpe ratio, reflecting superior risk-adjusted returns. This demonstrates that the agent was not only increasing returns but also managing risk effectively by adapting to market fluctuations.

Drawdown and Volatility: The DDPG portfolios experienced lower drawdowns and volatility, suggesting that the agent learned to avoid large losses during periods of negative price movements. This resilience was especially important in highly volatile market conditions, where the agent's continuous action space allowed for more adaptive trading decisions.

Reward Optimization: The model’s reward function, based on the percentage change in portfolio value, guided the agent to focus on maximizing overall portfolio growth rather than short-term profits. Over the course of training, the agent became increasingly effective at identifying when to hold positions versus when to reallocate resources, leading to higher long-term returns.

Adaptation to Market Conditions: The agent demonstrated a strong ability to adapt to the changing market dynamics within the simulation. For example, it learned to reallocate capital when certain stocks entered unfavorable trends, indicating that it was developing more advanced strategies beyond simple buying and holding. This adaptability is a key advantage of reinforcement learning models in portfolio optimization.

Identifying Bull Runs: One of its key strengths is its ability to swiftly recognize bull runs and dynamically shift portfolio weights towards high-performing stocks, capitalizing on positive market momentum. As you may have noticed, the algorithm really likes Nvidia. In fact, the algorithm really liked Nvidia on pretty much every test we ran it on for unsurprising reasons. This demonstrates the model's capability to identify market leaders and capitalize on them.
 
These findings suggest that using a DDPG agent in a continuous portfolio optimization setting provides great benefits. The section below highlights the potential for improvements when utilizing a q-learning-based model in a real-world trading environment. 

#### Limitation and Future work

Since we do not have access to real-time data, the training is done with simulated historical data. The code can be easily adopted to accommodate real-time data with the use of CSP.

The model assumes zero transaction costs, making it ideal for high-frequency rebalancing strategies. In real-world applications, transaction costs would need to be factored in for optimal performance.

One key area for improvement is to train on more iterations and portfolios. Training the Q-learning agent on a limited number of portfolios and over the period of a single trading day may lead to underfitting. Another area for improvement is expanding the observation space. Currently, the model only uses stock prices and portfolio holdings. Adding technical indicators like moving averages would provide deeper market insights and help the agent understand trends and overbought/oversold conditions more effectively.

Lastly, hyperparameter tuning remains an ongoing area of refinement and future work could implement more complex trading actions, including short selling, using leverage, or trading derivatives. This would enhance the realism of the simulations.

## Sample Output

Below is an example of the portfolio performance graph generated by the model:

![Portfolio Performance](/images/Portfolio_comparison.png)

This graph illustrates the training phase, where multiple portfolios are evaluated, followed by the testing phase where the best-performing portfolio is used.

## Conclusion

This project demonstrates the potential of using a reinforcement learning approach combined with the CSP data streaming framework for real-time portfolio optimization. By leveraging the CSP framework, we were able to train multiple portfolios simultaneously as the market data reveals itself, which allows the agent to adjust its portfolio to dynamic market conditions. Despite of using historical data in our simulations, the model has shown promising results in adapting to market movements. The findings suggest that the DDGG based agents can be highly effective in maximizing returns in a real-time trading environment. Further work can build on this foundation by incorporating real-time data and enhancing the model’s complexity with additional market indicators. Overall, this project shows the value of combining machine learning techniques with real-time data streaming for modern portfolio management.

## Getting Started

### How to Run the Portfolio Manager

Follow these steps to set up and run the portfolio_manager.py file:

1. **Set up a Python environment:**
   It's recommended to use a virtual environment to avoid conflicts with other projects. You can create one using:

   ```
   python -m venv portfolio_env
   ```

2. **Activate the virtual environment:**

   - On Windows:
     ```
     portfolio_env/Scripts/activate
     ```
   - On macOS and Linux:
     ```
     source portfolio_env/bin/activate
     ```

3. **Install the required dependencies:**
   With the virtual environment activated, use pip to install the dependencies listed in requirements.txt:

   ```
   pip install -r requirements.txt
   ```

   This will install all necessary packages, including numpy, pandas, matplotlib, tensorflow, csp, and tqdm.

4. **Prepare your data:**
   Ensure that your "HistoricalEquityData.csv" file is in the same directory as portfolio_manager.py.

5. **Run the portfolio manager:**
   Execute the portfolio_manager.py script:

   ```
   python portfolio_manager.py
   ```

6. **View the results:**
   The script will run the simulation, display a progress bar, and then show a plot of the portfolio performance. It will also print out key metrics like final portfolio value, total return, and Sharpe ratio.

### Troubleshooting

- If you encounter any "module not found" errors, ensure that all dependencies were installed correctly in step 3.
- Make sure your Python version is compatible with the requirements. This project is typically tested with Python 3.7-3.10.
- If you face issues with the CSP library, you might need to install it separately or check for any specific installation instructions on its repository.

### Notes

- The simulation may take some time to run, especially for large datasets. The progress bar will keep you informed of the simulation's progress.
- You can adjust parameters like INITIAL_BALANCE and TRAINING_STEPS in the portfolio_manager.py file to experiment with different scenarios.
- Remember that this simulation assumes zero transaction costs. In a real-world scenario, you would need to account for these costs.

## Dependencies

- numpy
- pandas
- matplotlib
- tensorflow
- csp
- tqdm

## Potential Improvements

- Integration with live market data feeds
- Incorporation of transaction costs for more realistic simulations
- Enhanced feature engineering for improved model performance

## Contributors

Colin White:

https://github.com/colinjw458

https://www.linkedin.com/in/colin-white-b66413189/


Mackenzie Qu:

https://github.com/mackenziequ
