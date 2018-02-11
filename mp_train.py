from time import sleep
import numpy as np
import multiprocessing as mp


def worker(weights, work_queue, result_queue):
    while True:
        seed = work_queue.get()
        if seed is None:
            work_queue.task_done()
            return
        gen = np.random.RandomState(seed)
        w = np.array(weights)
        delta = gen.randn(*w.shape)
        for i in range(100000):
            value = ((w + delta) ** 2).sum()
        result_queue.put((seed, value))
        work_queue.task_done()

if __name__ == '__main__':
    weights = mp.Array('d', [2] * 100)
    work_queue = mp.JoinableQueue()
    result_queue = mp.Queue()
    workers = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=worker, args=(weights, work_queue, result_queue))
        workers.append(p)

    for p in workers:
        p.start()

    best = float("inf")
    for _ in range(10):
        for i in range(100):
            work_queue.put(np.random.randint(1 << 32))

        work_queue.join()

        best_seed = None
        while not result_queue.empty():
            seed, value = result_queue.get()
            if value < best:
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

    print(w)
