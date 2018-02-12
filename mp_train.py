from time import sleep
import numpy as np
import multiprocessing as mp
from gym_puyopuyo.env import register

import tensorflow as tf
import keras.layers as kl
from keras import backend as K

from keras_agent import get_simple_agent, agent_performance
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def worker(global_weights, work_queue, result_queue):
    register()
    with tf.device("/cpu:0"):
         with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as session:
            K.set_session(session)
            agent = get_simple_agent(session, 4)
            session.run(tf.global_variables_initializer())

            variables = tf.trainable_variables()
            weigths = session.run(variables)
            w_shape = []
            placeholders = []
            assigns = []
            for values, variable in zip(weigths, variables):
                placeholders.append(tf.placeholder(tf.float32, values.shape))
                assigns.append(variable.assign(placeholders[-1]))
                w_shape.append(values.shape)

            while True:
                seed = work_queue.get()
                if seed is None:
                    work_queue.task_done()
                    return

                gen = np.random.RandomState(seed)
                w = np.array(global_weights)
                delta = gen.randn(*w.shape)
                w += delta * 10

                feed_dict = {}
                for shape, placeholder in zip(w_shape, placeholders):
                    values = w[:np.prod(shape)].reshape(shape)
                    feed_dict[placeholder] = values
                    w = w[np.prod(shape):]
                session.run(assigns, feed_dict=feed_dict)
                agent.reset()

                value = agent_performance(agent, 100)

                result_queue.put((seed, value))
                work_queue.task_done()


def get_initial_weights():
    register()
    with tf.device("/cpu:0"):
         with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as session:
            K.set_session(session)
            agent = get_simple_agent(session, 1)
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
    return w


if __name__ == '__main__':
    # w = get_initial_weights()
    # np.save("temp", w)
    # w = np.load("temp.npy")

    w = np.load("best.npy")

    weights = mp.Array('d', w)
    work_queue = mp.JoinableQueue()
    result_queue = mp.Queue()
    workers = []
    # num_workers = mp.cpu_count()
    num_workers = 8
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(weights, work_queue, result_queue))
        workers.append(p)

    for p in workers:
        p.start()

    npop = 32

    best = float("-inf")
    for _ in range(100):
        for i in range(npop):
            work_queue.put(np.random.randint(1 << 32))

        work_queue.join()

        best_seed = None
        while not result_queue.empty():
            seed, value = result_queue.get()
            if value > best:
                best = value
                best_seed = seed
        print(best, best_seed)
        if best_seed is None:
            continue

        gen = np.random.RandomState(best_seed)
        w = np.array(weights)
        delta = gen.randn(*w.shape)
        w += delta
        for i, weight in enumerate(w):
            weights[i] = weight

    for worker in workers:
        work_queue.put(None)
    for worker in workers:
        worker.join()

    np.save("best", w)
