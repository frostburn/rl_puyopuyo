from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict, deque
import os
import random
import sys

import gym
import numpy as np
import tensorflow as tf
from gym_puyopuyo.env import register
from gym_puyopuyo.util import print_up

from util import bias_variable, conv2d, summarize_scalar, variable_summaries, vh_log, weight_variable, parse_record, read_record


FLAGS = None


class Agent(object):
    BATCH_SIZE = 200
    KERNEL_SIZE = 5
    NUM_FEATURES = 30
    FC_1_SIZE = 1000
    FC_2_SIZE = 500

    def __init__(self, session, env):
        self.session = session
        self.env = env
        self.make_graph()
        self.make_summaries()
        if FLAGS:
            self.writer = tf.summary.FileWriter(FLAGS.log_dir)
            self.writer.add_graph(tf.get_default_graph())

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
        tf.summary.histogram("value_head", self.value_head)
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
            # flat_input = tf.reshape(self.box_input, [-1, self.n_box])
            # n_flat += self.n_box
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

        with tf.name_scope("fully_connected_2v"):
            self.W_fc_2_value = weight_variable([self.FC_1_SIZE, self.FC_2_SIZE], name="W")
            self.b_fc_2_value = bias_variable([self.FC_2_SIZE], name="b")
            z = tf.matmul(self.fc_1_activation, self.W_fc_2_value) + self.b_fc_2_value
            self.fc_2_activation_value = tf.sigmoid(z)

    def make_output_graph(self):
        self.n_actions = self.env.action_space.n
        with tf.name_scope("policy"):
            self.W_policy = weight_variable([self.FC_2_SIZE, self.n_actions], name="W")
            self.b_policy = bias_variable([self.n_actions], name="b")
            self.policy_head = tf.matmul(self.fc_2_activation_policy, self.W_policy) + self.b_policy
            self.policy_actions = tf.nn.softmax(logits=self.policy_head, dim=1, name="policy_actions")

        with tf.name_scope("value"):
            self.W_value = weight_variable([self.FC_2_SIZE, 1], name="W")
            self.b_value = bias_variable([1], name="b")
            self.value_head = tf.matmul(self.fc_2_activation_value, self.W_value) + self.b_value

    def make_loss_graph(self):
        with tf.name_scope("loss"):
            self.policy_target = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.n_actions], name="policy_target")
            self.value_target = tf.placeholder(tf.float32, [self.BATCH_SIZE, 1], name="value_target")
            with tf.name_scope("error"):
                self.loss_xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.policy_target, logits=self.policy_head)
                self.loss_mse = tf.reduce_mean(tf.squared_difference(self.value_head, self.value_target))
            with tf.name_scope("regularization"):
                regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
                reg_variables = tf.trainable_variables()
                self.reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            self.loss = self.loss_xent + self.loss_mse * 3e-4 + self.reg_term

    def make_train_graph(self):
        learning_rate = FLAGS.learning_rate if FLAGS else 0
        with tf.name_scope("train"):
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
            self.train_step = self.optimizer.minimize(self.loss)

    def get_feed_dict(self, experiences):
        feed_dict = defaultdict(list)
        for experience in experiences:
            (deal, box), action, value = experience
            feed_dict[self.deal_input].append(deal)
            feed_dict[self.box_input].append(box)
            feed_dict[self.policy_target].append(action)
            feed_dict[self.value_target].append([value])
        return feed_dict

    def get_policy_action(self, env, state):
        experiences = [(state, [0] * self.n_actions, 0)] * self.BATCH_SIZE
        feed_dict = self.get_feed_dict(experiences)
        action_dist = self.session.run(self.policy_actions, feed_dict=feed_dict)[0]
        action_mask = env.unwrapped.get_action_mask()
        action_dist *= action_mask
        if action_dist.any():
            action_dist /= action_dist.sum()
            return np.random.choice(self.n_actions, p=action_dist)
        else:
            return np.random.randint(0, self.n_actions)

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
            "W_fc_1", "b_fc_1",
            "W_fc_2_policy", "b_fc_2_policy",
            "W_fc_2_value", "b_fc_2_value",
            "W_policy", "b_policy",
            "W_value", "b_value",
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


def get_experiences(env):
    result = []
    path = "records/puyobot"
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as f:
            result.extend(read_record(env, f.read(), log_reward_scale=True))
    return result


def main(*args, **kwargs):
    with tf.Session() as session:
        env = gym.make("PuyoPuyoEndlessTsu-v0")
        agent = Agent(session, env)
        merged = tf.summary.merge_all()
        session.run(tf.global_variables_initializer())
        if FLAGS.params_dir:
            agent.load(FLAGS.params_dir)
        iteration = 0
        for episode in range(FLAGS.num_episodes):
            experiences = get_experiences(env)
            random.shuffle(experiences)
            while True:
                batch_size = agent.BATCH_SIZE // 4
                batch, experiences = experiences[:batch_size], experiences[batch_size:]
                for _ in range(3):
                    batch += [(env.permute_observation(o), a, v) for o, a, v in batch[:batch_size]]
                if len(batch) < agent.BATCH_SIZE:
                    break
                feed_dict = agent.get_feed_dict(batch)
                session.run(agent.train_step, feed_dict=feed_dict)
                if iteration % 100 == 0:
                    summary = session.run(merged, feed_dict=feed_dict)
                    agent.writer.add_summary(summary, iteration)
                iteration += 1

            total_reward = 0
            for k in range(100):
                state = env.reset()
                for j in range(1000):
                    state, reward, done, _ = env.step(agent.get_policy_action(env, state))
                    total_reward += reward
                    # env.render()
                    if done:
                        break
            print("Total reward =", total_reward)
            summarize_scalar(agent.writer, "policy_reward", total_reward, iteration)
            print("epoch done")
            agent.dump()
        agent.writer.close()


def main_with_render(*args, **kwargs):
    try:
        print("\033[?25l")
        main(*args, **kwargs)
    finally:
        print("\033[?25h")


if __name__ == "__main__":
    register()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of episodes to run the trainer")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps per episode")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--log_dir", type=str, default="/tmp/tensorflow/gym_puyopuyo/logs/rl_with_summaries",
                        help="Summaries log directory")
    parser.add_argument("--no_render", action="store_true",
                        help="Don't render visuals for episodes")
    parser.add_argument("--params_dir", type=str, default=None,
                        help="Parameters directory for initial values")
    parser.add_argument("--exploration", type=float, default=1.0,
                        help="Initial level of exploration for training")
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.do_render = not FLAGS.no_render
    main_fun = main_with_render if FLAGS.do_render else main
    FLAGS.use_convolution = True
    tf.app.run(main=main_fun, argv=[sys.argv[0]] + unparsed)
