# Stolen from https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d
# Thanks!

"""
A bare bones examples of optimizing a black-box function (f) using
Natural Evolution Strategies (NES), where the parameter distribution is a 
gaussian of fixed standard deviation.
"""

import argparse

# import numpy as np
# from keras_vs_agent import *
from keras_agent import *
from util import vh_log

# np.random.seed(0)

N_ENVS = 8
EPISODE_LENGTH = 16

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='Filename for outputs')
    parser.add_argument(
        '--input', type=str, default=None,
        help='A file containing initial parameters for the training',
    )
    args = parser.parse_args()

    register()
    with tf.Session() as session:
        K.set_session(session)

        log_dir = FLAGS.log_dir if FLAGS else "/tmp/tensorflow"
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(tf.get_default_graph())

        agent = get_simple_agent(session, N_ENVS)

        session.run(tf.global_variables_initializer())

        # Initial guess by Keras
        variables = tf.trainable_variables()
        weigths = session.run(variables)
        w = []
        w_shape = []
        placeholders = []
        assigns = []
        for values, variable in zip(weigths, variables):
            placeholders.append(tf.placeholder(tf.float32, values.shape))
            assigns.append(variable.assign(placeholders[-1]))
            w_shape.append(values.shape)
            w.extend(values.flatten())
        print(w_shape)
        w = np.array(w)

        # The function we want to optimize
        def f(w):
            feed_dict = {}
            for shape, placeholder in zip(w_shape, placeholders):
                values = w[:np.prod(shape)].reshape(shape)
                feed_dict[placeholder] = values
                w = w[np.prod(shape):]
            session.run(assigns, feed_dict=feed_dict)
            agent.reset()
            reward = agent_performance(agent, EPISODE_LENGTH)
            reg_term = float(np.dot(w.T, w))
            return reward - reg_term * 1e-5

        if args.input:
            print("loading params...")
            w = np.loadtxt(args.input, delimiter=",")

        # hyperparameters
        npop = 16 # population size
        sigma = 1e-2 # noise standard deviation
        alpha = 1e-5 # learning rate

        total_diff = np.zeros_like(w)
        last_diff = np.zeros_like(w)
        jittered_total = 0

        i = 0
        while i < 100 or True:
            i += 1
            # print current fitness of the most likely parameter setting
            if i % 10 == 0:
                vh_log({
                    "reward": f(w) / N_ENVS / EPISODE_LENGTH,
                    "jittered_reward": jittered_total / 10 / npop / N_ENVS / EPISODE_LENGTH,
                    "magnitude": float(np.dot(w.T, w)),
                    "change": float(np.dot(total_diff.T, total_diff)),
                    "direction": float(np.dot(total_diff.T, last_diff)),
                }, i)
                last_diff = total_diff
                total_diff = 0
                jittered_total = 0
            if i % 20 == 0 and True:
                filename = "/tmp/nes.csv"
                np.savetxt(filename, w, delimiter=",")
                print("Saved parameters to", filename)

            # initialize memory for a population of w's, and their rewards
            N = np.random.randn(npop, len(w)) # samples from a normal distribution N(0,1)
            R = np.zeros(npop)
            for j in range(npop):
                w_try = w + sigma * N[j]  # jitter w using gaussian of sigma 0.1
                R[j] = f(w_try)  # evaluate the jittered version
                jittered_total += R[j]

            std = np.std(R)
            # standardize the rewards to have a gaussian distribution
            A = (R - np.mean(R)) / (std + 1e-10 * (std == 0))
            # perform the parameter update. The matrix multiply below
            # is just an efficient way to sum up all the rows of the noise matrix N,
            # where each row N[j] is weighted by A[j]
            diff = alpha / (npop * sigma) * np.dot(N.T, A)
            w += diff

            agent.env.render()
            print(jittered_total)

            total_diff += diff

        writer.close()

        np.savetxt(args.output, w, delimiter=",")
        print("Saved parameters to", args.output)
