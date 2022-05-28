# Jimmy Ou Yang , 2030559
# R. Vincent , instructor
# Advanced Programming , section 1
# Final Project
# May 13 2022

from training import build_model, build_agent

"""
Passes player count, player count softness, and the dealer's upcard into the model,
and outputs the best move according to a set of trained weights.
"""

weights = input("Weights filepath (leave out file extension): ")  # .index file
model = build_model()
dqn = build_agent(model)
dqn.load_weights(weights)

actions = {0: "Stand", 1: "Hit", 2: "Double"}

try:
    while True:
        print("--------------")
        state = int(input("Player count: "))
        if state not in range(4, 20):
            raise ValueError("Player count must be between 4 and 20")
        soft = int(input("Is player count soft? (N=0, Y=1) "))
        if soft not in range(2):
            raise ValueError("Player count softness is either 0 or 1")
        dealer = int(input("Dealer upcard: "))
        if dealer not in range(2, 12):
            raise ValueError("Dealer upcard must be between 2 and 11")

        softness = "Soft" if soft == 1 else "Hard"
        action = dqn.forward([state, soft, dealer])
        print(f"{softness} {state} against {dealer}, you should {actions[action]}")

except ValueError as e:
    print(e)
