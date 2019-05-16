import torch


LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]


def get_action_space():
    return len(ACTIONS)


def get_action(q_value):
    action_index = q_value.max(1)[1]
    action = ACTIONS[action_index[0]]
    return action_index[0], action
