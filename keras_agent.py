import tensorflow as tf
import keras.layers as kl
from keras import backend as K
import numpy as np
import gym
from gym_puyopuyo.env import register

FLAGS = None


class SimpleAgent(object):
    def __init__(self, session, envs):
        self.session = session
        self.envs = envs
        self.env = envs[0]
        self.reset()
        self.make_graph()

    def make_graph(self):
        self.make_input_graph()

        self.dense_layers = []
        self.dense_layers.append(kl.Dense(128, activation='relu')(self.flat_input))
        self.dense_layers.append(kl.Dense(128, activation='relu')(self.dense_layers[-1]))

        self.make_output_graph()

    def make_input_graph(self):
        deal_space, box_space = self.env.observation_space.spaces
        with tf.name_scope("input"):
            self.deal_input = tf.placeholder(tf.float32, [None] + list(deal_space.shape), name="deal")
            self.box_input = tf.placeholder(tf.float32, [None] + list(box_space.shape), name="box")
        self.n_deal = np.prod(deal_space.shape)
        self.n_box = np.prod(box_space.shape)
        self.n_inputs = self.n_deal + self.n_box

        with tf.name_scope("flatten"):
            flat_input = tf.reshape(self.box_input, [-1, self.n_box])
            self.flat_input = tf.concat([flat_input, tf.reshape(self.deal_input, [-1, self.n_deal])], 1)

    def make_output_graph(self):
        self.n_actions = self.env.action_space.n
        self.policy_actions = kl.Dense(self.n_actions, activation='softmax')(self.dense_layers[-1])

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
            # action = np.random.choice(self.n_actions, p=dist)
            action = np.argmax(dist)
            # print(action)
            observation, reward, done, info = env.step(action)
            if done:
                observation = env.reset()
            self.observations.append(observation)
            states.append(info["state"])
            rewards.append(reward)
        return rewards

    def reset(self):
        for i, env in enumerate(self.envs):
            env.seed(i + 1235)
        self.observations = [env.reset() for env in self.envs]


def get_agent(session, batch_size):
    envs = [gym.make("PuyoPuyoEndlessSmall-v2") for _ in range(batch_size)]

    return SimpleAgent(session, envs)


def agent_performance(agent, episode_length):
    total = 0
    for _ in range(episode_length):
        total += sum(agent.step())
    return total


if __name__ == '__main__':
    register()
    with tf.Session() as session:
        K.set_session(session)

        log_dir = FLAGS.log_dir if FLAGS else "/tmp/tensorflow"
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(tf.get_default_graph())

        agent = get_agent(session, 10)

        session.run(tf.global_variables_initializer())


        for i in range(100):
            rewards = agent.step()
            agent.env.render()
            print(rewards)

        writer.close()
