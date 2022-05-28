# Jimmy Ou Yang , 2030559
# R. Vincent , instructor
# Advanced Programming , section 1
# Final Project
# May 13 2022

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from training import build_model, build_agent


def get_actions(weights):
    """Return arrays of actions for each possible hand"""
    model = build_model()
    dqn = build_agent(model)
    dqn.load_weights(weights)

    hard = []  # Hard hands
    for i in range(8, 20 + 1):  # Player cards
        row = []
        for j in range(2, 11 + 1):  # Dealer upcard
            action = dqn.forward([i, 0, j])  # Agent action
            row.append(action)
        hard.append(row)

    soft = []  # Soft hands
    for i in range(13, 20 + 1):  # Player cards
        row = []
        for j in range(2, 11 + 1):  # Dealer upcard
            action = dqn.forward([i, 1, j])  # Agent action
            row.append(action)
        soft.append(row)

    return (hard, soft)


def generate_graphic(actions, output):
    """Generates action chart for a set of actions"""

    hard, soft = actions
    n_hard = len(np.unique(hard))  # Counts unique actions
    n_soft = len(np.unique(soft))
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list("cmap", ["red", "green", "blue"], N=n_hard)
    fig = plt.figure(figsize=(5, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.title.set_text("HARD")
    cax1 = plt.pcolor(hard, cmap=color_map, edgecolor="#000000", linewidth=0.5)  # Sometimes, there is no doubles
    cbar = fig.colorbar(cax1)  # Legend
    cbar.set_ticks(range(0, 3))
    cbar.set_ticklabels(["STAND", "HIT", "DOUBLE"][:n_soft])
    ax1.set_xticks([float(x - 0.5) for x in range(1, 11)])  # Sets tick in middle of cell
    ax1.set_xticklabels(range(2, 12))
    ax1.xaxis.tick_top()

    ax1.set_yticks([float(x - 0.5) for x in range(1, 11)])  # Sets tick in middle of cell
    ax1.set_yticklabels(range(8, 18))

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.title.set_text("SOFT")
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list("a", ["red", "green", "blue"], N=n_soft)
    cax2 = plt.pcolor(soft, cmap=color_map, edgecolor="#000000", linewidth=0.5)
    cbar2 = fig.colorbar(cax2)  # Legend
    cbar2.set_ticks(range(n_soft))
    cbar2.set_ticklabels(["STAND", "HIT", "DOUBLE"][:n_soft])  # Sometimes, there is no doubles
    ax2.set_xticks([float(x - 0.5) for x in range(1, 11)])  # Sets tick in middle of cell
    ax2.set_xticklabels(range(2, 12))
    ax2.xaxis.tick_top()
    ax2.set_yticks([float(x - 0.5) for x in range(1, 9)])  # Sets tick in middle of cell
    ax2.set_yticklabels(range(13, 20 + 1))

    fig.savefig(f"{output}.png")
