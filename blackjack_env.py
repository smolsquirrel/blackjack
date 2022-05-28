# Jimmy Ou Yang , 2030559
# R. Vincent , instructor
# Advanced Programming , section 1
# Final Project
# May 13 2022

import numpy as np
import random
from gym import Env
from gym.spaces import Box, Discrete


class Blackjack(Env):
    """
    Basic implementation of a Blackjack environment, a classic casino game.
    Detailed rules can be found in the documentation.
    This environment is meant for training reinforcement learning models,
    not to be played by people.
    """

    SHOE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # Possible cards that can be drawn. J,Q,K count as 10

    class Agent:
        """Base class for agents, entities that can interact within the game"""

        def __init__(self):
            self.soft = False  # Hard/soft count
            self.state = 0  # Current sum of cards in hand

        def reset(self):
            """Reset hand"""
            self.card = 0
            self.soft = False
            self.state = 0

        def hit(self):
            """Receive an additional card from the shoe"""
            self.card = random.choice(Blackjack.SHOE)
            if self.card == 1:  # Special ace case
                self.__ace()
                state = self.state
            else:
                state = self.state + self.card  # Add new card to count

            if state > 21 and self.soft:  # Check for soft hand if count is over 21
                state -= 10  # Turn ace from 11 to 1
                self.soft = False

            self.state = state

        def __ace(self):
            """Handles the special case of an ace"""
            state = self.state + 11  # Start by trying highest value, 11
            if state > 21:  # If bust, turn ace to 1
                state -= 10
            else:
                self.soft = True  # Ace 11 turns count soft

            self.state = state

    class Player(Agent):
        """Player agent can decide their actions in the game"""

        def __init__(self):
            super().__init__()
            self.multiplier = 1  # Bet multiplier

        def reset(self):
            """Reset hand"""
            super().reset()
            self.multiplier = 1

    class Dealer(Agent):
        """Dealer agent can't make decisions. All actions are determined by rule. Acts as the house"""

        def __init__(self):
            super().__init__()
            self.downcard = 0  # Face down second card from initial deal
            self.bj = False  # Dealer blackjack is significant during training

        def reset(self):
            """Reset hand"""
            super().reset()
            self.downcard = 0

        def hit(self):
            """Receive an additional card from the shoe"""
            if self.state != 0 and self.downcard == 0:  # Second drawn card is face down and hidden from players
                self.card = random.choice(Blackjack.SHOE)
                if self.card == 1:  # Ace case
                    self.downcard = 11
                    self.soft = True
                else:
                    self.downcard = self.card
            else:
                super().hit()  # Normal hits, before and after the downcard is given

        def reveal(self):
            """Reveals the downcard and adds to count"""
            if self.state == 11 and self.downcard == 11:  # Ace case
                self.state = 12
                self.soft = False
            else:
                self.state += self.downcard
                if self.state == 21:  # Blackjack
                    self.bj = True

    def __init__(self):
        self.action_space = Discrete(3)  # 3 actions possible: STAND, HIT, DOUBLE
        # Player state, Player soft, Dealer stat
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([30, 1, 30]), dtype=np.int32)
        self.done = False
        self.player = Blackjack.Player()  # Create Player agent
        self.dealer = Blackjack.Dealer()  # Create Dealer agent

    def step(self, action):
        """Handles what happens when an action is taken"""
        info = {}
        if action == 0:  # Stand
            self.done = True  # End round
        elif action == 1:  # Hit
            self.player.hit()
        elif action == 2:  # Double
            self.player.hit()
            self.player.multiplier = 2  # Double bet/reward
            self.done = True  # End round

        if self.done or self.player.state >= 21:  # Rounded ended by player or by count (21 or bust)
            self.done = True
            self.dealer.reveal()
            while self.dealer.state < 17:  # Dealer keeps hitting until his count is 17 or above
                self.dealer.hit()
            if self.dealer.state == 17 and self.dealer.soft >= 1:  # Dealer hits soft 17
                self.dealer.hit()

            # Base reward is 10
            if self.player.state > 21:  # Bust, loss
                reward = -10 * self.player.multiplier
            elif self.dealer.bj:  # Dealer blackjacks, loss
                reward = -10 * self.player.multiplier
            else:
                if self.dealer.state > 21:  # Dealer busts
                    reward = 10 * self.player.multiplier
                else:
                    if self.player.state > self.dealer.state:  # Win
                        reward = 10 * self.player.multiplier
                    elif self.player.state == self.dealer.state:  # Push, bet is returned
                        reward = 0
                    else:
                        reward = -10 * self.player.multiplier  # Loss
        else:
            # Reward shaping, gives more reward the closer the count is to 21
            reward = 1 - ((21 - self.player.state) / 21) ** 0.4

        s = self.player.state, self.player.soft, self.dealer.state
        return s, reward, self.done, info

    def reset(self):
        """Resets the round"""
        self.player.reset()
        self.dealer.reset()
        self.player.hit()
        self.dealer.hit()
        self.player.hit()
        self.dealer.hit()
        if self.player.state == 21:  # Player blackjack requires no actions, ignored during training
            self.reset()

        self.done = False
        s = self.player.state, self.player.soft, self.dealer.state
        return s
