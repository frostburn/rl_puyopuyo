from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import gym
import numpy as np
import tensorflow as tf
from gym_puyopuyo.agent import SmallTreeSearchAgent
from gym_puyopuyo.env import register
from gym_puyopuyo.util import print_up

from util import bias_variable, summarize_scalar, variable_summaries, vh_log, weight_variable  # noqa: I001


FLAGS = None


class Agent(object):
    BATCH_SIZE = 20
    FC_1_SIZE = 300
    FC_2_SIZE = 200

    def __init__(self, session, envs):
        self.session = session
        self.envs = envs
        self.observations = [env.reset() for env in self.envs]
        self.env = envs[0]
        self.make_graph()
        self.make_summaries()
        self.writer = tf.summary.FileWriter(FLAGS.log_dir)
        self.writer.add_graph(tf.get_default_graph())

    def make_graph(self):
        self.make_input_graph()
        self.make_hidden_graphs()
        self.make_output_graph()
        self.make_loss_graph()
        self.make_train_graph()

    def make_summaries(self):
        variable_summaries(self.W_hidden, "W_hidden")
        variable_summaries(self.b_hidden, "b_hidden")
        variable_summaries(self.W_hidden_2, "W_hidden_2")
        variable_summaries(self.b_hidden_2, "b_hidden_2")
        variable_summaries(self.W_policy, "W_policy")
        variable_summaries(self.b_policy, "b_policy")

        tf.summary.histogram('policy_head', self.policy_head)

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

    def make_hidden_graphs(self):
        with tf.name_scope("flatten"):
            flat_input = tf.reshape(self.box_input, [-1, self.n_box])
            flat_input = tf.concat([flat_input, tf.reshape(self.deal_input, [-1, self.n_deal])], 1)

        with tf.name_scope("hidden"):
            self.W_hidden = weight_variable([self.n_inputs, self.FC_1_SIZE], name="W")
            self.b_hidden = bias_variable([self.FC_1_SIZE], name="b")
            z = tf.matmul(flat_input, self.W_hidden) + self.b_hidden
            self.hidden_1_activation = tf.sigmoid(z)

        with tf.name_scope("hidden_2"):
            self.W_hidden_2 = weight_variable([self.FC_1_SIZE, self.FC_2_SIZE], name="W")
            self.b_hidden_2 = bias_variable([self.FC_2_SIZE], name="b")
            z = tf.matmul(self.hidden_1_activation, self.W_hidden_2) + self.b_hidden_2
            self.hidden_activation = tf.sigmoid(z)

    def make_output_graph(self):
        self.n_actions = self.env.action_space.n
        with tf.name_scope("policy"):
            self.W_policy = weight_variable([self.FC_2_SIZE, self.n_actions], name="W")
            self.b_policy = bias_variable([self.n_actions], name="b")
            self.policy_head = tf.matmul(self.hidden_activation, self.W_policy) + self.b_policy
            self.policy_actions = tf.nn.softmax(logits=self.policy_head, dim=1, name="policy_actions")

    def make_loss_graph(self):
        with tf.name_scope("loss"):
            self.policy_target = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.n_actions], name="policy_target")
            with tf.name_scope("error"):
                self.loss_xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.policy_target, logits=self.policy_head)
            with tf.name_scope("regularization"):
                regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
                reg_variables = tf.trainable_variables()
                self.reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            self.loss = self.loss_xent + self.reg_term

    def make_train_graph(self):
        with tf.name_scope("train"):
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9, use_nesterov=True)
            # self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

    def get_feed_dict(self):
        feed_dict = {self.deal_input: [], self.box_input: []}
        for observation in self.observations:
            deal, box = observation
            feed_dict[self.deal_input].append(deal)
            feed_dict[self.box_input].append(box)
        return feed_dict

    def step(self):
        action_dists = self.session.run(self.policy_actions, feed_dict=self.get_feed_dict())
        self.observations = []
        rewards = []
        states = []
        for env, dist in zip(self.envs, action_dists):
            action = np.random.choice(self.n_actions, p=dist)
            observation, reward, done, info = env.step(action)
            if done:
                observation = env.reset()
            self.observations.append(observation)
            states.append(info["state"])
            rewards.append(reward)
        agent = SmallTreeSearchAgent(returns_distribution=True)
        targets = [agent.get_action(state) for state in states]
        feed_dict = self.get_feed_dict()
        feed_dict[self.policy_target] = targets
        self.session.run(self.train_step, feed_dict=feed_dict)
        return rewards, feed_dict

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
            "W_hidden", "b_hidden",
            "W_hidden_2", "b_hidden_2",
            "W_policy", "b_policy",
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
        envs = [gym.make("PuyoPuyoEndlessSmall-v1") for _ in range(Agent.BATCH_SIZE)]
        agent = Agent(session, envs)
        merged = tf.summary.merge_all()
        session.run(tf.global_variables_initializer())
        if FLAGS.params_dir:
            agent.load(FLAGS.params_dir)
        for iteration in range(FLAGS.num_iterations):
            rewards, feed_dict = agent.step()
            agent.envs[0].render()
            print(rewards[0], sum(rewards))
            summary = session.run(merged, feed_dict=feed_dict)
            agent.writer.add_summary(summary, iteration)
            summarize_scalar(agent.writer, "policy_reward", sum(rewards), iteration)
            if iteration % 100 == 0:
                agent.dump()
        agent.writer.close()


if __name__ == "__main__":
    register()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=10000,
                        help='Number of steps to run the trainer')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='Initial learning rate')
    parser.add_argument("--params_dir", type=str, default=None,
                        help="Parameters directory for initial values")
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/gym_puyopuyo/logs/rl_with_summaries',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
