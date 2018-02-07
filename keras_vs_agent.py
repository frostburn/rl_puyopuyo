import tensorflow as tf
import keras.layers as kl
from keras import backend as K
import numpy as np
import gym
from gym_puyopuyo.env import register, ENV_PARAMS
from gym_puyopuyo.env.versus import PuyoPuyoVersusEnv

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
        self.box_inputs = []
        self.n_other_inputs = 0
        with tf.name_scope("input"):
            for name, dict_space in zip(["my", "your"], self.env.observation_space.spaces):
                space = dict_space.spaces
                box_space = space["field"]
                self.box_shape = box_space.shape
                self.box_inputs.append(tf.placeholder(tf.float32, [None] + list(self.box_shape), name=name + "_box"))
                deal_space = space["deals"]
                n_deal = np.prod(deal_space.shape)
                self.n_other_inputs += n_deal
                self.n_other_inputs += len(space) - 2
            self.other_inputs = tf.placeholder(tf.float32, (None, self.n_other_inputs), name="other")
        self.n_box = np.prod(self.box_shape)
        self.n_inputs = self.n_box + self.n_other_inputs

    def get_feed_dict(self):
        feed_dict = {self.other_inputs: [], self.box_inputs[0]: [], self.box_inputs[1]: []}
        for observation in self.observations:
            feed_dict[self.box_inputs[0]].append(observation[0]["field"])
            feed_dict[self.box_inputs[1]].append(observation[1]["field"])
            other = []
            for obs in observation:
                other.extend(obs["deals"].flatten())
                other.append(obs["chain_number"])
                other.append(obs["pending_score"])
                other.append(obs["pending_garbage"])
                other.append(obs["all_clear"])
            feed_dict[self.other_inputs].append(other)
        return feed_dict

    def step(self, sampled=True):
        feed_dict = self.get_feed_dict()
        if sampled:
            action_dists = self.session.run(self.policy_actions, feed_dict=feed_dict)
        else:
            action_dists = self.session.run(self.policy_head, feed_dict=feed_dict)
        # self.observations = []
        rewards = []
        states = []
        for i, env, dist in zip(range(len(self.envs)), self.envs, action_dists):
            if env.unwrapped.state.game_over:
                rewards.append(0)
                continue
            if sampled:
                action = np.random.choice(self.n_actions, p=dist)
            else:
                action = np.argmax(dist)
            observation, reward, done, info = env.step(action)
            self.observations[i] = observation
            states.append(info["state"])
            rewards.append(reward)
        return rewards

    def reset(self):
        self.observations = [env.reset() for env in self.envs]


class SimpleAgent(BaseAgent):
    def make_hidden_graph(self):
        self.conv_layers_me = []
        with tf.name_scope("me"):
            self.conv_layers_me.append(kl.Conv2D(8, (3, 3), activation='relu', padding='valid')(self.box_inputs[0]))
        self.conv_layers_you = []
        with tf.name_scope("you"):
            self.conv_layers_you.append(kl.Conv2D(8, (3, 3), activation='relu', padding='valid')(self.box_inputs[1]))
        with tf.name_scope("flatten"):
            self.flat_input = kl.concatenate([
                kl.Flatten()(self.conv_layers_me[-1]),
                kl.Flatten()(self.conv_layers_you[-1]),
                self.other_inputs
                ], axis=-1
            )

        self.dense_layers = []
        self.dense_layers.append(kl.Dense(16, activation='relu')(self.flat_input))
        # self.dense_layers.append(kl.Dense(16, activation='relu')(self.dense_layers[-1]))

    def make_output_graph(self):
        self.n_actions = self.env.action_space.n
        self.policy_head = kl.Dense(self.n_actions)(self.dense_layers[-1])
        self.policy_actions = kl.Activation('softmax')(self.policy_head)


class DeepAgent(BaseAgent):
    def make_hidden_graph(self):
        convolvers = [
            kl.Conv2D(16, (3, 3), activation='relu', padding='valid'),
            kl.Conv2D(16, (3, 3), activation='relu', padding='valid'),
        ]
        self.conv_layers_me = []
        with tf.name_scope("me"):
            self.conv_layers_me.append(convolvers[0](self.box_inputs[0]))
            self.conv_layers_me.append(convolvers[1](self.conv_layers_me[-1]))
        self.conv_layers_you = []
        with tf.name_scope("you"):
            self.conv_layers_you.append(convolvers[0](self.box_inputs[1]))
            self.conv_layers_you.append(convolvers[1](self.conv_layers_you[-1]))
        with tf.name_scope("flatten"):
            self.flat_input = kl.concatenate([
                kl.Flatten()(self.conv_layers_me[-1]),
                kl.Flatten()(self.conv_layers_you[-1]),
                self.other_inputs
                ], axis=-1
            )

        self.dense_layers = []
        self.dense_layers.append(kl.Dense(64, activation='relu')(self.flat_input))
        self.dense_layers.append(kl.Dense(32, activation='relu')(self.dense_layers[-1]))

    def make_output_graph(self):
        self.n_actions = self.env.action_space.n
        self.policy_head = kl.Dense(self.n_actions)(self.dense_layers[-1])
        self.policy_actions = kl.Activation('softmax')(self.policy_head)


def get_simple_agent(session, batch_size):
    envs = [gym.make("PuyoPuyoVersusSmallEasy-v0") for _ in range(batch_size)]
    return SimpleAgent(session, envs)


def get_deep_agent(session, batch_size):
    envs = []
    for _ in range(batch_size):
        agent = lambda game: np.random.randint(0, 22)
        env = PuyoPuyoVersusEnv(agent, ENV_PARAMS["PuyoPuyoVersusTsu-v0"], garbage_clue_weight=1e-9)
        envs.append(env)
    return DeepAgent(session, envs)


def agent_performance(agent, episode_length):
    total = 0
    for _ in range(episode_length):
        rewards = agent.step()
        total += sum(rewards)
    return total


if __name__ == '__main__':
    register()
    with tf.device("/cpu:0"):
        with tf.Session() as session:
            K.set_session(session)

            agent = get_simple_agent(session, 10)

            log_dir = FLAGS.log_dir if FLAGS else "/tmp/tensorflow/keras_agent"
            writer = tf.summary.FileWriter(log_dir)
            writer.add_graph(tf.get_default_graph())

            session.run(tf.global_variables_initializer())

            for i in range(100):
                rewards = agent.step()
                agent.env.render()
                print(rewards)

            writer.close()
