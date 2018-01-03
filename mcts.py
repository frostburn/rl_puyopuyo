from __future__ import division

from collections import deque
import json
import pickle
import random
import sys

import tensorflow as tf
import numpy as np
import gym
from gym_puyopuyo.env import register

from util import GAMMA
from train_deep import Agent

EXPLORATION = 3
PLAYOUT_LENGTH = 50


class RandomAgent(object):
    BATCH_SIZE = 3

    def __init__(self, _, env):
        self.env = env

    def get_actions(self, states):
        result = []
        for state in states:
            dist = state.get_action_mask()
            if dist.any():
                dist /= dist.sum()
                result.append(np.random.choice(self.env.action_space.n, p=dist))
            else:
                result.append(np.random.randint(0, self.env.action_space.n))
        return result


class AgentWrapper(Agent):
    def __init__(self, *args):
        super().__init__(*args)
        self.load("/tmp/tensorflow/gym_puyopuyo/outputs/")

    def get_actions(self, states):
        frames = [state.encode() for state in states]
        action_dist = self.get_policy_dist(frames)
        result = []
        for state, dist in zip(states, action_dist):
            dist *= state.get_action_mask()
            if dist.any():
                dist /= dist.sum()
                result.append(np.random.choice(self.n_actions, p=dist))
            else:
                result.append(np.random.randint(0, self.n_actions))
        return result

class Node(object):
    def __init__(self, state, reward=0, action=None):
        self.state = state
        self.reward = reward
        self.action = action
        self.score = 0
        self.visits = 1
        self.children = []

    def expand(self):
        for action, (child, reward) in enumerate(self.state.get_children(True)):
            if child is None:
                continue
            self.children.append(Node(child, reward, action))

    def choose(self, exploration):
        best_value = float("-inf")
        best_child = None
        random.shuffle(self.children)
        for child in self.children:
            value = child.value / child.visits
            value += exploration * np.sqrt(np.log(self.visits) / child.visits)
            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    def render(self):
        self.children.sort(key=lambda child: child.action)
        self.state.render()
        print("{} / {} = {}".format(self.score, self.visits, self.score / self.visits))
        for child in self.children:
            print("  {} / {} = {}".format(child.value, child.visits, child.value / child.visits))

    @property
    def value(self):
        return self.score + self.reward

    @property
    def confidence(self):
        best_child = self.choose(0)
        return best_child.visits / self.visits


def playout(agent, state):
    states = [state.clone() for _ in range(agent.BATCH_SIZE)]
    rewards = [[] for _ in range(agent.BATCH_SIZE)]
    for _ in range(PLAYOUT_LENGTH):
        actions = agent.get_actions(states)
        for i in range(agent.BATCH_SIZE):
            state = states[i]
            rs = rewards[i]
            if rs and rs[-1] < 0:
                continue
            action = actions[i]
            reward = state.step(*state.actions[action])
            if reward >= 0:
                reward *= reward
            rs.append(reward)

    total_score = 0
    for rs in rewards:
        score = 0
        for reward in reversed(rs):
            score = reward + GAMMA * score
        total_score += score
    return score, agent.BATCH_SIZE


def mc_iterate(agent, node, exploration=EXPLORATION):
    path = [node]
    while node.children:
        node = node.choose(exploration)
        path.append(node)
    node.expand()
    score, visits = playout(agent, node.state)

    for node in reversed(path):
        node.score += score
        node.visits += visits
        score = node.reward + score * GAMMA


def mcts(agent_class):
    """
    Does a Monte Carlo tree search using an agent for rollout policy
    """
    with tf.Session() as session:
    # if True:
    #     session = None
        seed = random.randint(0, 1234567890)
        env = gym.make("PuyoPuyoEndlessSmall-v0")
        agent = agent_class(session, env)
        env.seed(seed)
        env.reset()
        root = Node(env.unwrapped.get_root())

        exploration = float(sys.argv[1])
        print("Exploration", exploration)

        with open("mcts_exploration_{}.record".format(exploration), "w") as f:
            f.write(str(seed) + "\n")
            while True:
                for _ in range(100):
                    mc_iterate(agent, root, exploration)
                i = 0
                while root.confidence < 0.2 and i < 5:
                    i += 1
                    print("Gaining more confidence... {} %".format(100 * root.confidence))
                    for _ in range(30):
                        mc_iterate(agent, root, exploration)
                root.render()
                action = root.choose(0).action
                f.write(str(action) + "\n")
                _, reward, done, _ = env.step(action)
                print("Reward =", reward)
                root = Node(env.unwrapped.get_root())
                if done:
                    break

if __name__ == "__main__":
    register()
    mcts(AgentWrapper)
    # mcts(RandomAgent)
