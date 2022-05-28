# Jimmy Ou Yang , 2030559
# R. Vincent , instructor
# Advanced Programming , section 1
# Final Project
# May 13 2022

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from blackjack_env import Blackjack

"""
Train the reinforcement learning model using an agent
"""

tf.get_logger().setLevel("ERROR")  # Disable warnings

env = Blackjack()  # Creates environment instance

actions = env.action_space.n  # Number of actions


# 4-layer neural network
# 3 nodes in input layer, Player state, Player soft, Dealer state (upcard)
# 2 hidden layers, 24 nodes each
# 3 nodes in output layer for 3 actions, STAND, HIT, DOUBLE
def build_model():
    """Creates the neural network model"""
    model = Sequential()
    model.add(Flatten(input_shape=(1, 3)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model):
    """Creates RL agent that will interact with environment"""
    # Policy for determining action
    # Eps is better for exploration
    # Boltzmann uses "correct" actions
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2
    )  # Agent that will interact with environment

    dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
    return dqn


if __name__ == "__main__":
    model = build_model()
    dqn = build_agent(model)

    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=1)  # Train model

    dqn.save_weights("a")
