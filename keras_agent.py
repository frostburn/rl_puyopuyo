import tensorflow as tf
import keras.layers as kl
from keras import backend as K
import numpy as np
import gym
from gym_puyopuyo.env import register

FLAGS = None


class BaseAgent(object):
    def __init__(self, session, envs):
        self.session = session
        self.envs = envs
        self.env = envs[0]
        self.reset()
        self.make_graph()

    def make_graph(self):
        self.make_input_graph()
        self.make_hidden_graph()
        self.make_output_graph()

    def make_input_graph(self):
        deal_space, box_space = self.env.observation_space.spaces
        self.box_shape = box_space.shape
        with tf.name_scope("input"):
            self.deal_input = tf.placeholder(tf.float32, [None] + list(deal_space.shape), name="deal")
            self.box_input = tf.placeholder(tf.float32, [None] + list(box_space.shape), name="box")
        self.n_deal = np.prod(deal_space.shape)
        self.n_box = np.prod(box_space.shape)
        self.n_inputs = self.n_deal + self.n_box

    def get_feed_dict(self):
        feed_dict = {self.deal_input: [], self.box_input: []}
        for observation in self.observations:
            deal, box = observation
            feed_dict[self.deal_input].append(deal)
            feed_dict[self.box_input].append(box)
        return feed_dict

    def step(self, sampled=False):
        feed_dict = self.get_feed_dict()
        if sampled:
            action_dists = self.session.run(self.policy_actions, feed_dict=feed_dict)
        else:
            action_dists = self.session.run(self.policy_head, feed_dict=feed_dict)
        self.observations = []
        rewards = []
        states = []
        for env, dist in zip(self.envs, action_dists):
            if sampled:
                action = np.random.choice(self.n_actions, p=dist)
            else:
                action = np.argmax(dist)
            observation, reward, done, info = env.step(action)
            if done:
                observation = env.reset()
            self.observations.append(observation)
            states.append(info["state"])
            rewards.append(reward)
        return rewards

    def reset(self):
        self.observations = [env.reset() for env in self.envs]


class SimpleAgent(BaseAgent):
    def make_hidden_graph(self):
        with tf.name_scope("flatten"):
            self.flat_input = kl.concatenate([
                kl.Flatten()(self.box_input),
                kl.Flatten()(self.deal_input)
                ], axis=-1
            )

        self.dense_layers = []
        self.dense_layers.append(kl.Dense(32, activation='relu')(self.flat_input))
        self.dense_layers.append(kl.Dense(16, activation='relu')(self.dense_layers[-1]))

    def make_output_graph(self):
        self.n_actions = self.env.action_space.n
        self.policy_head = kl.Dense(self.n_actions)(self.dense_layers[-1])
        self.policy_actions = kl.Activation('softmax')(self.policy_head)


class DeepAgent(BaseAgent):
    def make_hidden_graph(self):
        self.conv_layers = []
        self.conv_layers.append(kl.Conv2D(8, (3, 3), activation='relu', padding='same')(self.box_input))
        self.conv_layers.append(kl.Conv2D(8, (3, 3), activation='relu')(self.conv_layers[-1]))

        with tf.name_scope("flatten"):
            self.flat_input = kl.concatenate([
                kl.Flatten()(self.conv_layers[-1]),
                kl.Flatten()(self.deal_input)
                ], axis=-1
            )

        self.dense_layers = []
        self.dense_layers.append(kl.Dense(64, activation='relu')(self.flat_input))
        # self.dense_layers.append(kl.Dense(256, activation='relu')(self.dense_layers[-1]))

    def make_output_graph(self):
        self.n_actions = self.env.action_space.n
        self.policy_head = kl.Dense(self.n_actions)(self.dense_layers[-1])
        self.policy_actions = kl.Activation('softmax')(self.policy_head)


def get_simple_agent(session, batch_size):
    envs = [gym.make("PuyoPuyoEndlessSmall-v2") for _ in range(batch_size)]
    return SimpleAgent(session, envs)


def get_deep_agent(session, batch_size):
    envs = [gym.make("PuyoPuyoEndlessTsu-v2") for _ in range(batch_size)]
    return DeepAgent(session, envs)


def agent_performance(agent, episode_length, reshape=None):
    total = 0
    for _ in range(episode_length):
        rewards = agent.step()
        if reshape:
            rewards = reshape(rewards)
        total += sum(rewards)
    return total


if __name__ == '__main__':
    register()
    with tf.device("/cpu:0"):
        with tf.Session() as session:
            K.set_session(session)

            agent = get_deep_agent(session, 10)

            log_dir = FLAGS.log_dir if FLAGS else "/tmp/tensorflow/keras_agent"
            writer = tf.summary.FileWriter(log_dir)
            writer.add_graph(tf.get_default_graph())

            session.run(tf.global_variables_initializer())

            for i in range(100):
                rewards = agent.step()
                agent.env.render()
                print(rewards)

            writer.close()
