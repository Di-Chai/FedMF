import time
import copy
import numpy as np

from load_data import ratings_dict, item_id_list, user_id_list

hidden_dim = 100

user_vector = np.random.normal(size=[len(user_id_list), hidden_dim])

item_vector = np.random.normal(size=[len(item_id_list), hidden_dim])

max_iteration = 1000

reg_u = 1e-4
reg_v = 1e-4

lr = 1e-3

def user_update(single_user_vector, user_rating_list, item_vector):
    gradient = np.zeros([len(item_vector), len(single_user_vector)])
    for item_id, rate in user_rating_list:
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        single_user_vector = single_user_vector - lr * (-2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = error * single_user_vector
    return single_user_vector, gradient


def server_update(gradient):
    for g in gradient:
        item_vector = item_vector - lr * (-2 * g + 2 * reg_u * item_vector)


def iterate():
    gradient_all = []
    for i in range(len(user_id_list)):
        user_vector[i], gradient = user_update(user_vector[i], ratings_dict[user_id_list[i]], item_vector)
        gradient_all.append(gradient)
    server_update(gradient_all)


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
        
        t = time.time()
        iterate()
        print('Time', time.time() - t, 's')

        print('loss', loss())