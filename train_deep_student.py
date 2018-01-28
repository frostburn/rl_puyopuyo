from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict, deque
import json
import os
import random
import sys

import gym
import numpy as np
import tensorflow as tf
from gym_puyopuyo.agent import TsuTreeSearchAgent
from gym_puyopuyo.env import register
from gym_puyopuyo.util import print_up

from util import bias_variable, conv2d, summarize_scalar, variable_summaries, vh_log, weight_variable, parse_record, read_record, GAMMA


FLAGS = None
HYPERPARAMS = {
    "batch_size": 32,
    "augmentation": 3,
    "kernel_size": 5,
    "num_features": 2,
    "fc_1_size": 6,
    "fc_2_size": 3,
    "teacher_depth": 2,
}


class Agent(object):
    def __init__(self, session, envs):
        self.session = session
        self.envs = envs
        self.observations = [env.reset() for env in self.envs] * (1 + HYPERPARAMS["augmentation"])
        self.states = [env.unwrapped.get_root() for env in self.envs] * (1 + HYPERPARAMS["augmentation"])
        self.env = envs[0]
        self.make_graph()
        self.make_summaries()
        if FLAGS:
            self.writer = tf.summary.FileWriter(FLAGS.log_dir)
            self.writer.add_graph(tf.get_default_graph())

    @property
    def BATCH_SIZE(self):
        return HYPERPARAMS["batch_size"] * (1 + HYPERPARAMS["augmentation"])

    @property
    def KERNEL_SIZE(self):
        return HYPERPARAMS["kernel_size"]

    @property
    def NUM_FEATURES(self):
        return HYPERPARAMS["num_features"]

    @property
    def FC_1_SIZE(self):
        return HYPERPARAMS["fc_1_size"]

    @property
    def FC_2_SIZE(self):
        return HYPERPARAMS["fc_2_size"]

    def make_graph(self):
        self.make_input_graph()
        self.make_convolution_graph()
        self.make_fc_1_graph()
        self.make_fc_2_graph()
        self.make_output_graph()
        self.make_loss_graph()
        self.make_train_graph()

    def make_summaries(self):
        for variable, name in zip(self.variables, self.variable_names):
            variable_summaries(variable, name)
        tf.summary.histogram("policy_head", self.policy_head)
        tf.summary.histogram("Q_head", self.Q_head)
        tf.summary.scalar('loss_mse', tf.reduce_mean(self.loss_mse))
        tf.summary.scalar('loss_xent', tf.reduce_mean(self.loss_xent))
        tf.summary.scalar('loss', tf.reduce_mean(self.loss))

    def make_input_graph(self):
        deal_space, box_space = self.env.observation_space.spaces
        with tf.name_scope("input"):
            self.deal_input = tf.placeholder(tf.float32, [self.BATCH_SIZE] + list(deal_space.shape), name="deal")
            self.box_input = tf.placeholder(tf.float32, [self.BATCH_SIZE] + list(box_space.shape), name="box")
        self.n_deal = np.prod(deal_space.shape)
        self.n_box = np.prod(box_space.shape)
        self.n_inputs = self.n_deal + self.n_box
        self.box_shape = box_space.shape

    def make_convolution_graph(self):
        with tf.name_scope("convolution"):
            self.W_conv = weight_variable([self.KERNEL_SIZE, self.KERNEL_SIZE, self.box_shape[-1], self.NUM_FEATURES], name="W")
            self.b_conv = bias_variable([self.NUM_FEATURES], name="b")
            z = conv2d(self.box_input, self.W_conv) + self.b_conv
            self.box_activation = tf.sigmoid(z)
            self.n_conv = self.box_shape[0] * self.box_shape[1] * self.NUM_FEATURES

    def make_fc_1_graph(self):
        n_flat = 0
        with tf.name_scope("flatten"):
            flat_input = tf.reshape(self.box_activation, [-1, self.n_conv])
            n_flat += self.n_conv
            flat_input = tf.concat([flat_input, tf.reshape(self.deal_input, [-1, self.n_deal])], 1)
            n_flat += self.n_deal
        with tf.name_scope("fully_connected_1"):
            self.W_fc_1 = weight_variable([n_flat, self.FC_1_SIZE], name="W")
            self.b_fc_1 = bias_variable([self.FC_1_SIZE], name="b")
            z = tf.matmul(flat_input, self.W_fc_1) + self.b_fc_1
            self.fc_1_activation = tf.sigmoid(z)

    def make_fc_2_graph(self):
        with tf.name_scope("fully_connected_2p"):
            self.W_fc_2_policy = weight_variable([self.FC_1_SIZE, self.FC_2_SIZE], name="W")
            self.b_fc_2_policy = bias_variable([self.FC_2_SIZE], name="b")
            z = tf.matmul(self.fc_1_activation, self.W_fc_2_policy) + self.b_fc_2_policy
            self.fc_2_activation_policy = tf.sigmoid(z)

        with tf.name_scope("fully_connected_2Q"):
            self.W_fc_2_Q = weight_variable([self.FC_1_SIZE, self.FC_2_SIZE], name="W")
            self.b_fc_2_Q = bias_variable([self.FC_2_SIZE], name="b")
            z = tf.matmul(self.fc_1_activation, self.W_fc_2_Q) + self.b_fc_2_Q
            self.fc_2_activation_Q = tf.sigmoid(z)

    def make_output_graph(self):
        self.n_actions = self.env.action_space.n
        with tf.name_scope("policy"):
            self.W_policy = weight_variable([self.FC_2_SIZE, self.n_actions], name="W")
            self.b_policy = bias_variable([self.n_actions], name="b")
            self.policy_head = tf.matmul(self.fc_2_activation_policy, self.W_policy) + self.b_policy
            self.policy_actions = tf.nn.softmax(logits=5*self.policy_head, dim=1, name="policy_actions")

        with tf.name_scope("Q"):
            self.W_Q = weight_variable([self.FC_2_SIZE, self.n_actions], name="W")
            self.b_Q = bias_variable([self.n_actions], name="b")
            self.Q_head = tf.matmul(self.fc_2_activation_Q, self.W_Q) + self.b_Q

    def make_loss_graph(self):
        with tf.name_scope("loss"):
            self.policy_target = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.n_actions], name="policy_target")
            self.Q_target = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.n_actions], name="Q_target")
            with tf.name_scope("error"):
                self.loss_xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.policy_target, logits=self.policy_head)
                self.loss_mse = tf.reduce_mean(tf.squared_difference(self.Q_head, self.Q_target))
            with tf.name_scope("regularization"):
                regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
                reg_variables = tf.trainable_variables()
                self.reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            self.loss = self.loss_xent + self.loss_mse + self.reg_term

    def make_train_graph(self):
        learning_rate = FLAGS.learning_rate if FLAGS else 0
        with tf.name_scope("train"):
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
            self.train_step = self.optimizer.minimize(self.loss)

    def get_feed_dict(self, observations=None):
        observations = observations or self.observations
        feed_dict = {self.deal_input: [], self.box_input: []}
        for observation in observations:
            deal, box = observation
            feed_dict[self.deal_input].append(deal)
            feed_dict[self.box_input].append(box)
        return feed_dict

    def get_policy_targets(self, states=None):
        states = states or self.states
        agent = TsuTreeSearchAgent(returns_distribution=True)
        agent.depth = HYPERPARAMS["teacher_depth"]
        return [agent.get_action(state) for state in states]

    def get_Q_targets(self, Q_base, actions, observations, rewards):
        feed_dict = self.get_feed_dict(observations)
        new_Qs = self.session.run(self.Q_head, feed_dict=feed_dict)
        targets = []
        for target, responsible_action, reward, Q in zip(Q_base, actions, rewards, new_Qs):
            target[responsible_action] = reward + GAMMA * np.max(Q)
            targets.append(target)
        return targets

    def step(self):
        feed_dict = self.get_feed_dict()
        action_dists, Q_base = self.session.run((self.policy_actions, self.Q_head), feed_dict=feed_dict)
        # print(action_dists[0])
        actions = []
        observations = []
        rewards = []
        states = []
        for env, dist in zip(self.envs, action_dists):
            action = np.random.choice(self.n_actions, p=dist)
            observation, reward, done, info = env.step(action)
            reward = np.cbrt(reward)
            if done:
                observation = env.reset()
            for i in range(1 + HYPERPARAMS["augmentation"]):
                if i > 0:
                    observation = self.env.permute_observation(observation)
                actions.append(action)
                observations.append(observation)
                states.append(info["state"])
                rewards.append(reward)

        feed_dict[self.policy_target] = self.get_policy_targets()
        feed_dict[self.Q_target] = self.get_Q_targets(Q_base, actions, observations, rewards)
        self.session.run(self.train_step, feed_dict=feed_dict)

        self.observations = observations
        self.states = states

        return rewards, feed_dict

    def get_policy_dist(self, states):
        experiences = list((state, [0] * self.n_actions, 0) for state in states)
        feed_dict = self.get_feed_dict(experiences)
        return self.session.run(self.policy_actions, feed_dict=feed_dict)

    def render_in_place(self):
        self.env.render()
        print_up(8)

    def render_ansi(self):
        sio = self.env.render("ansi")
        print_up(8, outfile=sio)
        return sio

    @property
    def variable_names(self):
        return [
            "W_conv", "b_conv",
            "W_fc_1", "b_fc_1",
            "W_fc_2_policy", "b_fc_2_policy",
            "W_fc_2_Q", "b_fc_2_Q",
            "W_policy", "b_policy",
            "W_Q", "b_Q",
        ]

    @property
    def variables(self):
        return [getattr(self, name) for name in self.variable_names]

    def dump(self):
        outputs_dir = os.getenv("VH_OUTPUTS_DIR", "/tmp/tensorflow/gym_puyopuyo/outputs")
        if not os.path.isdir(outputs_dir):
            os.makedirs(outputs_dir)
        arrays = self.session.run(self.variables)
        for arr, name in zip(arrays, self.variable_names):
            arr = arr.flatten()
            filename = os.path.join(outputs_dir, "{}.csv".format(name))
            np.savetxt(filename, arr, delimiter=",")
        print("Saved parameters to {}".format(outputs_dir))

    def load(self, params_dir):
        for variable, name in zip(self.variables, self.variable_names):
            filename = os.path.join(params_dir, "{}.csv".format(name))
            arr = np.loadtxt(filename, delimiter=",")
            arr = arr.reshape(variable.shape)
            self.session.run(variable.assign(arr))
        print("Loaded parameters from {}".format(params_dir))


def main(*args, **kwargs):
    with tf.Session() as session:
        envs = [gym.make("PuyoPuyoEndlessTsu-v2") for _ in range(HYPERPARAMS["batch_size"])]
        agent = Agent(session, envs)
        merged = tf.summary.merge_all()
        session.run(tf.global_variables_initializer())
        running_reward = 0
        if FLAGS.params_dir:
            agent.load(FLAGS.params_dir)
        for iteration in range(FLAGS.num_iterations):
            rewards, feed_dict = agent.step()
            if not FLAGS.quiet:
                agent.envs[0].render()
                print(rewards[0])
            running_reward += sum(rewards)
            if iteration % 10 == 0:
                vh_log({"reward": running_reward}, iteration)
                if not FLAGS.quiet:
                    summarize_scalar(agent.writer, "reward", running_reward, iteration)
                running_reward = 0
            if iteration % 100 == 0 and not FLAGS.quiet:
                summary = session.run(merged, feed_dict=feed_dict)
                agent.writer.add_summary(summary, iteration)
                agent.dump()
        agent.dump()
        agent.writer.close()


if __name__ == "__main__":
    register()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=10000,
                        help='Number of steps to run the trainer')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument("--params_dir", type=str, default=None,
                        help="Parameters directory for initial values")
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/gym_puyopuyo/logs/rl_with_summaries',
                        help='Summaries log directory')
    parser.add_argument('--hyperparams', type=str, default='{}',
                        help='Hyperparameters (JSON or filename)')
    parser.add_argument('--quiet', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    try:
        hyperparams = json.loads(FLAGS.hyperparams)
    except ValueError:
        with open(FLAGS.hyperparams) as f:
            hyperparams = json.load(f)
    HYPERPARAMS.update(hyperparams)
    print(HYPERPARAMS)
    print("Iterations =", FLAGS.num_iterations)
    print("Learning rate =", FLAGS.learning_rate)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
