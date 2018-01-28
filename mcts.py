from __future__ import division

import argparse
from collections import deque
import json
import pickle
import random
import sys
import os

import tensorflow as tf
import numpy as np
import gym
from gym_puyopuyo.agent import SmallTreeSearchAgent
from gym_puyopuyo.env import register

from util import GAMMA, vh_log, get_or_create_outputs_dir
from train_student import Agent

EXPLORATION = 3
PLAYOUT_LENGTH = 25

FLAGS = None


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
    BATCH_SIZE = 3

    def __init__(self, *args):
        super().__init__(*args)
        # self.load("/tmp/tensorflow/gym_puyopuyo/outputs/")
        self.load("params/student_100_50/")

    def get_actions(self, states):
        self.observations = [state.encode() for state in states]
        action_dists = self.session.run(self.policy_actions, feed_dict=self.get_feed_dict())
        result = []
        for state, dist in zip(states, action_dists):
            dist *= state.get_action_mask()
            if dist.any():
                dist /= dist.sum()
                result.append(np.random.choice(self.n_actions, p=dist))
            else:
                result.append(np.random.randint(0, self.n_actions))
        return result


class TreeAgent(SmallTreeSearchAgent):
    depth = 3
    BATCH_SIZE = 1

    def __init__(self, *args):
        super().__init__()

    def get_actions(self, states):
        return [self.get_action(state) for state in states]


class Node(object):
    def __init__(self, state, reward=0, action=None, depth=None):
        self.state = state
        self.reward = reward
        self.action = action
        self.depth = depth
        self.score = 0
        self.visits = 0
        self.children = []

    def expand(self):
        if self.depth == 0:
            return
        for action, (child, reward) in enumerate(self.state.get_children(True)):
            if child is None:
                continue
            depth = None if self.depth is None else self.depth - 1
            self.children.append(Node(child, reward, action, depth))

    def choose(self, exploration, state=None):
        best_value = float("-inf")
        best_child = None
        random.shuffle(self.children)
        for child in self.children:
            if not child.visits:
                best_child = child
                break
            value = child.value / child.visits
            value += exploration * np.sqrt(np.log(self.visits) / child.visits)
            if value > best_value:
                best_value = value
                best_child = child
        if state is not None:
            state.step(*state.actions[best_child.action])
        return best_child

    def render(self):
        self.children.sort(key=lambda child: child.action)
        self.state.render()
        print("{} / {} = {}".format(self.score, self.visits, self.score / self.visits if self.visits else None))
        for child in self.children:
            print("  {} / {} = {}".format(child.value, child.visits, child.value / child.visits if child.visits else None))

    @property
    def value(self):
        return self.score + self.reward

    @property
    def confidence(self):
        best_child = self.choose(0)
        if best_child is None:
            return 0
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
            rs.append(reward)

    total_score = 0
    for rs in rewards:
        score = 0
        for reward in reversed(rs):
            score = reward + GAMMA * score
        total_score += score
    return score, agent.BATCH_SIZE


def mc_iterate(agent, node, exploration=EXPLORATION):
    state = node.state.clone()
    path = [node]
    while node.children:
        node = node.choose(exploration, state)
        path.append(node)
    node.expand()
    score, visits = playout(agent, state)

    for node in reversed(path):
        node.score += score
        node.visits += visits
        score = node.reward + score * GAMMA


def mcts(agent_class):
    """
    Does a Monte Carlo tree search using an agent for rollout policy
    """
    with tf.Session() as session:
        seed = random.randint(0, 1234567890)
        env = gym.make("PuyoPuyoEndlessSmall-v1")
        agent = agent_class(session, [env] * agent_class.BATCH_SIZE)
        env.seed(seed)
        env.reset()
        state = env.unwrapped.get_root()
        root = Node(state, depth=state.num_deals)

        exploration = FLAGS.exploration
        print("Exploration =", exploration)

        outputs_dir = get_or_create_outputs_dir()
        with open(os.path.join(outputs_dir, "mcts_tsu_exploration_{}.record".format(exploration)), "w") as f:
            f.write(str(seed) + "\n")
            total_reward = 0
            for iteration in range(FLAGS.num_steps):
                for _ in range(200):
                    mc_iterate(agent, root, exploration)
                i = 0
                while root.confidence < 0.2 and i < 5:
                    i += 1
                    print("Gaining more confidence... {} %".format(100 * root.confidence))
                    for _ in range(15):
                        mc_iterate(agent, root, exploration)
                root.render()
                action = root.choose(0).action
                f.write(str(action) + "\n")
                _, reward, done, _ = env.step(action)
                total_reward += reward
                vh_log({
                    "reward": reward,
                    "total_reward": total_reward,
                    "average_reward": total_reward / (iteration + 1),
                }, iteration)
                state = env.unwrapped.get_root()
                root = Node(state, depth=state.num_deals)
                if done:
                    break

if __name__ == "__main__":
    register()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Max Number of steps")
    parser.add_argument("--exploration", type=float, default=8,
                        help="Level of exploration")
    FLAGS, unparsed = parser.parse_known_args()
    # mcts(AgentWrapper)
    # mcts(RandomAgent)
    mcts(TreeAgent)
