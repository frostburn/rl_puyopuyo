from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf

GAMMA = 0.95


def weight_variable(shape, name, stddev=None):
    """
    Create a weight variable with appropriate initialization.
    Defaults to Xavier initialization.
    """
    if stddev is None:
        stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(stddev=stddev, shape=shape)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name, value=0.0):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(input, kernel):
    """Returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME')


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries-{}'.format(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def summarize_scalar(writer, tag, value, step):
    """Add a custom summary outside of the main graph."""
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


def vh_log(data, step):
    """Log data for valohai."""
    data["step"] = step
    print(json.dumps(data))


def rewards_to_values(rewards):
    values = []
    value = 0
    for reward in reversed(rewards):
        value = reward + GAMMA * value
        values.insert(0, value)
    return values


def one_hot_record(env, observations, actions, rewards, done):
    values = rewards_to_values(rewards)

    result = []
    stuff = list(zip(observations, actions, values))
    if not done:
        stuff = stuff[:-50]
    for observation, action, value in stuff:
        action_one_hot = np.zeros(env.action_space.n)
        action_one_hot[action] = 1
        yield (observation, action_one_hot, value)


def parse_record(env, lines):
    """
    Parses a seed + action record into a trainable sequence
    """
    lines = list(map(int, lines))
    seed = lines[0]
    actions = lines[1:]

    env.seed(seed)
    env.reset()
    observations = []
    rewards = []
    for action in actions:
        # env.render()
        observation, reward, done, _ = env.step(action)
        observations.append(observation)
        rewards.append(reward)

    for item in one_hot_record(env, observations, actions, rewards, done):
        yield item


def read_record(env, file):
    observations = []
    rewards = []
    actions = []
    for observation, reward, done, info in env.read_record(file):
        observations.append(observation)
        rewards.append(reward)
        actions.append(info["action"])

    for item in one_hot_record(env, observations, actions, rewards, done):
        yield item


def get_or_create_outputs_dir():
    outputs_dir = os.getenv("VH_OUTPUTS_DIR", "/tmp/tensorflow/gym_puyopuyo/outputs")
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)
    return outputs_dir
