from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import gym
import numpy as np
import tensorflow as tf
from gym_puyopuyo.env import register
from gym_puyopuyo.util import print_up

from util import bias_variable, summarize_scalar, variable_summaries, vh_log, weight_variable  # noqa: I001


FLAGS = None


class Agent(object):
    BATCH_SIZE = 1
    HIDDEN_SIZE = 200

    def __init__(self):
        self.env = gym.make('PuyoPuyoEndlessSmall-v0')
        self.make_graph()
        self.make_summaries()
        self.writer = tf.summary.FileWriter(FLAGS.log_dir)
        self.writer.add_graph(tf.get_default_graph())

    def make_graph(self):
        self.make_input_graph()
        self.make_hidden_graph()
        self.make_output_graph()
        self.make_loss_graph()
        self.make_train_graph()

    def make_summaries(self):
        variable_summaries(self.W_hidden, "W_hidden")
        variable_summaries(self.b_hidden, "b_hidden")
        variable_summaries(self.W_output, "W_output")
        variable_summaries(self.b_output, "b_output")

        tf.summary.histogram('Q', self.output)
        tf.summary.histogram('action', self.action_dist)

    def make_input_graph(self):
        deal_space, box_space = self.env.observation_space.spaces
        with tf.name_scope("input"):
            self.deal_input = tf.placeholder(tf.float32, [self.BATCH_SIZE] + list(deal_space.shape), name="deal")
            self.box_input = tf.placeholder(tf.float32, [self.BATCH_SIZE] + list(box_space.shape), name="box")
        self.n_deal = np.prod(deal_space.shape)
        self.n_box = np.prod(box_space.shape)
        self.n_inputs = self.n_deal + self.n_box

    def make_hidden_graph(self):
        with tf.name_scope("flatten"):
            flat_input = tf.reshape(self.box_input, [-1, self.n_box])
            flat_input = tf.concat([flat_input, tf.reshape(self.deal_input, [-1, self.n_deal])], 1)

        with tf.name_scope("hidden"):
            self.W_hidden = weight_variable([self.n_inputs, self.HIDDEN_SIZE], name="W")
            self.b_hidden = bias_variable([self.HIDDEN_SIZE], name="b")
            z = tf.matmul(flat_input, self.W_hidden) + self.b_hidden
            self.hidden_activation = tf.sigmoid(z)

    def make_output_graph(self):
        self.n_outputs = self.env.action_space.n
        with tf.name_scope("output"):
            self.W_output = weight_variable([self.HIDDEN_SIZE, self.n_outputs], name="W")
            self.b_output = bias_variable([self.n_outputs], name="b")
            z = tf.matmul(self.hidden_activation, self.W_output) + self.b_output
            self.output = z
            self.action_dist = tf.nn.softmax(logits=z*3, dim=1, name="actions")
            # self.action = tf.argmax(self.output, 1)

    def make_loss_graph(self):
        with tf.name_scope("loss"):
            self.target = tf.placeholder(tf.float32, [self.BATCH_SIZE, self.n_outputs], name="target")
            with tf.name_scope("error"):
                self.error = tf.reduce_sum(tf.square(self.output - self.target))
            with tf.name_scope("L2-norm"):
                self.L2_norm = tf.reduce_sum(tf.square(self.b_hidden)) + tf.reduce_sum(tf.square(self.b_output))
                self.L2_norm += tf.reduce_sum(tf.square(self.W_hidden)) + tf.reduce_sum(tf.square(self.W_output))
            self.loss = self.error + self.L2_norm * 1e-9

    def make_train_graph(self):
        with tf.name_scope("train"):
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9, use_nesterov=True)
            # self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

    def get_feed_dict(self, state):
        deal, box = state
        return {self.deal_input: [deal], self.box_input: [box]}

    def render_in_place(self):
        self.env.render()
        print_up(8)

    def render_ansi(self):
        sio = self.env.render("ansi")
        print_up(8, outfile=sio)
        return sio

    @property
    def variables(self):
        return [self.W_hidden, self.b_hidden, self.W_output, self.b_output]

    def dump(self, session):
        outputs_dir = os.getenv('VH_OUTPUTS_DIR', '/tmp/tensorflow/gym_puyopuyo/outputs')
        if not os.path.isdir(outputs_dir):
            os.makedirs(outputs_dir)
        arrays = session.run(self.variables)
        for arr, name in zip(arrays, ["W_hidden", "b_hidden", "W_output", "b_output"]):
            filename = os.path.join(outputs_dir, "{}.csv".format(name))
            np.savetxt(filename, arr, delimiter=",")
        print("Saved parameters to {}".format(outputs_dir))

    def load(self, session, params_dir):
        for variable, name in zip(self.variables, ["W_hidden", "b_hidden", "W_output", "b_output"]):
            filename = os.path.join(params_dir, "{}.csv".format(name))
            arr = np.loadtxt(filename, delimiter=",")
            session.run(variable.assign(arr))
        print("Loaded parameters from {}".format(params_dir))


def main(*args, **kwargs):
    gamma = 0.99
    agent = Agent()
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.params_dir:
            agent.load(sess, FLAGS.params_dir)
        for i in range(FLAGS.num_episodes):
            exploration = FLAGS.exploration / (0.1 * i + 2)
            vh_log({"exploration": exploration}, i)
            total_reward = 0
            state = agent.env.reset()
            frames = []
            for j in range(FLAGS.num_steps):
                action_dist, Q_base = sess.run([agent.action_dist, agent.output], feed_dict=agent.get_feed_dict(state))
                action_dist *= agent.env.unwrapped.get_action_mask()
                if np.random.rand(1) < exploration or not action_dist.any():
                    action = agent.env.action_space.sample()
                else:
                    action_dist /= action_dist.sum()
                    action = np.random.choice(agent.n_outputs, p=action_dist[0])
                new_state, reward, done, _ = agent.env.step(action)
                Q = sess.run(agent.output, feed_dict=agent.get_feed_dict(new_state))  # noqa: N806
                Q_target = Q_base  # noqa: N806
                Q_target[0, action] = reward + gamma * np.max(Q)
                if FLAGS.do_render and False:
                    # We only render 10% from the start of the episode so as not to clog the console.
                    frames.append(agent.render_ansi().getvalue())
                    if j % 10 == 0:
                        print(frames.pop(0), end="")
                feed_dict = agent.get_feed_dict(state)
                feed_dict[agent.target] = Q_target
                sess.run(agent.train_step, feed_dict=feed_dict)

                total_reward += reward
                state = new_state
                if done:
                    break
            if FLAGS.do_render:
                agent.env.render()
            vh_log({"episode_length": j}, i)
            vh_log({"reward": total_reward}, i)
            summarize_scalar(agent.writer, "Steps", j, i)
            summarize_scalar(agent.writer, "Reward", total_reward, i)
            summary = sess.run(merged, feed_dict=feed_dict)
            agent.writer.add_summary(summary, i)
        agent.dump(sess)
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
    parser.add_argument('--num_episodes', type=int, default=10000,
                        help='Number of episodes to run the trainer')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='Number of steps per episode')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/gym_puyopuyo/logs/rl_with_summaries',
                        help='Summaries log directory')
    parser.add_argument('--no_render', action='store_true',
                        help="Don't render visuals for episodes")
    parser.add_argument('--params_dir', type=str, default=None,
                        help='Parameters directory for initial values')
    parser.add_argument('--exploration', type=float, default=1.0,
                        help='Initial level of exploration for training')
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.do_render = not FLAGS.no_render
    main_fun = main_with_render if FLAGS.do_render else main
    tf.app.run(main=main_fun, argv=[sys.argv[0]] + unparsed)
