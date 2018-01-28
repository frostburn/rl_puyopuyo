# Stolen from https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d
# Thanks!

"""
A bare bones examples of optimizing a black-box function (f) using
Natural Evolution Strategies (NES), where the parameter distribution is a 
gaussian of fixed standard deviation.
"""

# import numpy as np
from keras_agent import *
from util import vh_log

np.random.seed(0)

if __name__ == '__main__':
    register()
    with tf.Session() as session:
        K.set_session(session)

        log_dir = FLAGS.log_dir if FLAGS else "/tmp/tensorflow"
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(tf.get_default_graph())

        agent = get_agent(session, 8)

        session.run(tf.global_variables_initializer())

        # Initial guess from Keras
        # w = agent.get_params()

        variables = tf.trainable_variables()
        weigths = session.run(variables)
        w = []
        w_shape = []
        for values in weigths:
            w_shape.append(values.shape)
            w.extend(values.flatten())
        print(w_shape)
        w = np.array(w)

        # the function we want to optimize
        def f(w):
            assigns = []
            for shape, variable in zip(w_shape, variables):
                values = w[:np.prod(shape)].reshape(shape)
                assigns.append(variable.assign(values))
                # feed_dict[variable] = values
            session.run(assigns)
            agent.reset()
            reward = agent_performance(agent, 16)
            return reward

        # hyperparameters
        npop = 50 # population size
        sigma = 1e-1 # noise standard deviation
        alpha = 1e-2 # learning rate

        last_diff = np.zeros_like(w)
        jittered_total = 0

        for i in range(1000):
            # print current fitness of the most likely parameter setting
            if i % 10 == 0:
                vh_log({"reward": f(w), "jittered_reward": jittered_total}, i)
                jittered_total = 0

            # initialize memory for a population of w's, and their rewards
            N = np.random.randn(npop, len(w)) # samples from a normal distribution N(0,1)
            R = np.zeros(npop)
            for j in range(npop):
                w_try = w + sigma * N[j]  # jitter w using gaussian of sigma 0.1
                R[j] = f(w_try)  # evaluate the jittered version
                jittered_total += R[j]

            std = np.std(R)
            if not std.any():
                continue
            # standardize the rewards to have a gaussian distribution
            A = (R - np.mean(R)) / std
            if not A.any():
                continue
            # perform the parameter update. The matrix multiply below
            # is just an efficient way to sum up all the rows of the noise matrix N,
            # where each row N[j] is weighted by A[j]
            diff = alpha/(npop*sigma) * np.dot(N.T, A)
            w = w + diff

            vh_log({"magnitude": np.dot(w.T, w), "change": np.dot(diff.T, diff), "direction": np.dot(diff.T, last_diff)}, i)
            last_diff = diff

        writer.close()
