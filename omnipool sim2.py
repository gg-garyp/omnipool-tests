# Version 1.13

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(filename='simulation_logs.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

num_tokens = 10
initial_prices = np.ones(num_tokens)
pool_weights = np.full(num_tokens, 0.1)

def min_max_scaling(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def update_weights(pool_weights, liquidity, volume, volatility, market_cap):
    liquidity_weight = liquidity / np.sum(liquidity)
    volume_weight = volume / np.sum(volume)
    volatility_weight = 1 - (volatility / np.sum(volatility))
    market_cap_weight = market_cap / np.sum(market_cap)

    total_weight = liquidity_weight + volume_weight + volatility_weight + market_cap_weight
    new_weights = (pool_weights / np.sum(pool_weights)) * total_weight

    # constrain new_weights to be between 0.1 and 2.0
    new_weights = np.clip(new_weights, 0.1, 2.0)

    return new_weights

def simulate_swap(token_1_index, token_2_index, token_1_amount, pool_prices, pool_weights, token_0_value, token_volumes):
    token_1_price = pool_prices[token_1_index]
    token_2_price = pool_prices[token_2_index]

    token_1_volume = token_volumes[token_1_index]
    token_2_volume = token_volumes[token_2_index]

    # Calculate the effective token values
    token_1_effective_value = token_1_volume * token_1_price / token_0_value
    token_2_effective_value = token_2_volume * token_2_price / token_0_value

    # Calculate the amount of token 2 that needs to be swapped for token 1
    token_2_amount = token_1_amount * (token_1_effective_value / token_2_effective_value)

    # Update the token volumes
    token_volumes[token_1_index] -= token_1_amount
    token_volumes[token_2_index] += token_2_amount

    # Calculate the new prices and weights
    pool_prices[token_1_index] = token_1_volume / (token_volumes[token_1_index] * token_0_value)
    pool_prices[token_2_index] = token_2_volume / (token_volumes[token_2_index] * token_0_value)

    pool_weights = pool_prices * token_volumes / np.sum(pool_prices * token_volumes)

    return pool_prices, pool_weights, token_volumes

liquidity = min_max_scaling(np.random.rand(num_tokens) * 1000)
volume = min_max_scaling(np.random.rand(num_tokens) * 1000)
volatility = min_max_scaling(np.random.rand(num_tokens) * 1000)
market_cap = min_max_scaling(np.random.rand(num_tokens) * 1000)
pool_prices = np.ones(num_tokens)

token_0_value = pool_prices[0]

num_iterations = 750

token_values = []
token_volumes = np.zeros((num_iterations, num_tokens))

for i in range(num_iterations):
    pool_weights = update_weights(pool_weights, liquidity, volume, volatility, market_cap)

    # Choose two random tokens to swap between
    token_indices = np.random.choice(num_tokens, size=2, replace=False)

    # Compute the normalized weights for the two tokens
    normalized_sell_weight = pool_weights[token_indices[0]] / np.sum(pool_weights)
    normalized_buy_weight = pool_weights[token_indices[1]] / np.sum(pool_weights)

    # Compute the sell price based on the normalized weights and the XYK constant
    sell_price = pool_prices[token_indices[0]] * (1 - (normalized_sell_weight / XYK) ** 2)

    # Compute the buy price based on the normalized weights and the XYK constant
    buy_price = pool_prices[token_indices[1]] * (1 + (normalized_buy_weight / XYK) ** 2)

    # Compute the amount of tokens to sell and buy based on the ratio of the two prices
    amount_to_sell = np.random.rand() * pool_weights[token_indices[0]]
    amount_to_buy = amount_to_sell * sell_price / buy_price

    # Update the token volumes and weights after the swap
    token_volumes[i, token_indices[0]] -= amount_to_sell
    token_volumes[i, token_indices[1]] += amount_to_buy
    pool_weights[token_indices[0]] -= amount_to_sell
    pool_weights[token_indices[1]] += amount_to_buy

    # Compute the new prices for all tokens based on the updated weights and volumes
    pool_prices = np.zeros(num_tokens)
    for j in range(num_tokens):
        if pool_weights[j] == 0:
            pool_prices[j] = 0
        else:
            pool_prices[j] = np.sum(token_volumes[i] * pool_weights) / (pool_weights[j] * num_tokens)

    # Round pool_weights and pool_prices to 4 decimal places
    pool_weights = np.round(pool_weights, 4)
    pool_prices = np.round(pool_prices, 4)

    df = pd.DataFrame({
        "pool_weights": pool_weights,
        "pool_prices": pool_prices
    })

    dirname = f"pool_states/run_{i}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    df.to_csv(f"{dirname}/pool_state_{i}.csv")

    logging.info(f"Iteration {i}:")
    logging.info(f"  Num tokens: {num_tokens}")
    logging.info(f" High pool weight: {np.round(np.max(pool_weights), 4)}")
    logging.info(f" Low pool weight: {np.round(np.min(pool_weights), 4)}")
    logging.info(f" Avg pool weight: {np.round(np.mean(pool_weights), 4)}")

    for j in range(num_tokens):
        token_volume = token_volumes[i, j]
        logging.info(f"  Token {j+1} volume: {np.round(token_volume, 4)}")

    total_pool_value = np.sum(pool_weights * pool_prices)
    logging.info(f" Total pool value: {np.round(total_pool_value, 4)}")

    plt.subplot(2, 1, 1)
    for j in range(num_tokens):
        token_value = pool_weights[j] * pool_prices[j]
        plt.plot(range(i+1), token_volumes[:i+1, j] * token_value, label=f"Token {j+1}")
        plt.ylabel("Token Value")
    plt.legend()

    total_pool_value = np.sum(pool_weights * pool_prices)
    axs[1].plot(range(i+1), total_pool_value, color='blue')
    axs[1].set_ylabel("Total Pool Value")

    plt.xlabel("Iteration")
    plt.show()

    logging.info(f"Iteration {i}:")
    logging.info(f"  Num tokens: {num_tokens}")
    logging.info(f" High pool weight: {np.round(np.max(pool_weights), 4)}")
    logging.info(f" Low pool weight: {np.round(np.min(pool_weights), 4)}")
    logging.info(f" Avg pool weight: {np.round(np.mean(pool_weights), 4)}")

    for j in range(num_tokens):
        token_volume = token_volumes[i, j]
        logging.info(f"  Token {j+1} volume: {np.round(token_volume, 4)}")

    total_pool_value = np.sum(pool_weights * pool_prices)
    logging.info(f" Total pool value: {np.round(total_pool_value, 4)}")


