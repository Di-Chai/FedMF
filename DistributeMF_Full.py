import time
import copy
import numpy as np

from load_data import ratings_dict, item_id_list, user_id_list
from shared_parameter import *


def user_update(single_user_vector, user_rating_list, item_vector):
    gradient = np.zeros([len(item_vector), len(single_user_vector)])
    for item_id, rate in user_rating_list:
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        single_user_vector = single_user_vector - lr * (-2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = error * single_user_vector
    return single_user_vector, gradient


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

    # Init process

    user_vector = np.random.normal(size=[len(user_id_list), hidden_dim])

    item_vector = np.random.normal(size=[len(item_id_list), hidden_dim])

    start_time = time.time()

    for iteration in range(max_iteration):

        print('###################')
        t = time.time()

        # Step 2 User updates
        gradient_from_user = []
        for i in range(len(user_id_list)):
            user_vector[i], gradient = user_update(user_vector[i], ratings_dict[user_id_list[i]], item_vector)
            gradient_from_user.append(gradient)

        # Step 3 Server update
        tmp_item_vector = copy.deepcopy(item_vector)
        for g in gradient_from_user:
            item_vector = item_vector - lr * (-2 * g + 2 * reg_u * item_vector)

        if np.mean(np.abs(item_vector - tmp_item_vector)) < 1e-4:
            print('Converged')
            break

        print('Time', time.time() - t, 's')

        print('loss', loss())

    print('Converged using', time.time() - start_time)