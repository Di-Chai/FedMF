import time
import copy
import numpy as np

from numba import jit
from load_data import ratings_dict, item_id_list, user_id_list

hidden_dim = 100

user_vector = np.random.normal(size=[len(user_id_list), hidden_dim])

item_vector = np.random.normal(size=[len(item_id_list), hidden_dim])

max_iteration = 1000

reg_u = 1e-4
reg_v = 1e-4

lr = 1e-3


def iterate():
    # User updates
    for i in range(len(user_id_list)):

        for r in range(len(ratings_dict[user_id_list[i]])):

            item_id, rate = ratings_dict[user_id_list[i]][r]

            error = rate - np.dot(user_vector[i], item_vector[item_id])

            user_vector[i] = user_vector[i] - \
                             lr * (-2 * error * item_vector[item_id] + 2 * reg_u * user_vector[i])

            item_vector[item_id] = item_vector[item_id] - \
                                   lr * (-2 * error * user_vector[i] + 2 * reg_v * item_vector[item_id])


def loss():
    loss = []
    # User updates
    for i in range(len(user_id_list)):
        for r in range(len(ratings_dict[user_id_list[i]])):
            item_id, rate = ratings_dict[user_id_list[i]][r]
            error = (rate - np.dot(user_vector[i], item_vector[item_id])) ** 2
            loss.append(error)

    return np.mean(loss)


if __name__ == '__main__':

    for iteration in range(max_iteration):

        print('#################################')

        tmp_user_vector = copy.deepcopy(user_vector)
        tmp_item_vector = copy.deepcopy(item_vector)

        t = time.time()
        iterate()
        print('Time', time.time() - t, 's')

        print('loss', loss())

        if np.mean(np.abs(user_vector - tmp_user_vector)) < 1e-4 and\
           np.mean(np.abs(item_vector - tmp_item_vector)) < 1e-4:
            print('Converged')