import sys
import time
import copy
import numpy as np

from pympler import asizeof
from phe import paillier
from shared_parameter import *
from load_data import ratings_dict, item_id_list, user_id_list


def user_update(single_user_vector, user_rating_list, encrypted_item_vector):

    item_vector = np.array([[private_key.decrypt(e) for e in vector] for vector in encrypted_item_vector],
                           dtype=np.float32)

    gradient = np.zeros([len(item_vector), len(single_user_vector)])
    for item_id, rate in user_rating_list:
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        single_user_vector = single_user_vector - lr * (-2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = lr * (-2 * error * single_user_vector + 2 * reg_v * item_vector[item_id])

    encrypted_gradient = [[public_key.encrypt(e, precision=1e-5) for e in vector] for vector in gradient]

    return single_user_vector, encrypted_gradient


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

    # Step 1 Server encrypt item-vector
    t = time.time()
    encrypted_item_vector = [[public_key.encrypt(e, precision=1e-5) for e in vector] for vector in item_vector]
    print('Item profile encrypt using', time.time() - t, 'seconds')

    for iteration in range(max_iteration):

        print('###################')
        t = time.time()

        # Step 2 User updates
        cache_size = (sys.getsizeof(encrypted_item_vector[0][0].ciphertext()) +
                      sys.getsizeof(encrypted_item_vector[0][0].exponent)) *\
                     len(encrypted_item_vector) *\
                     len(encrypted_item_vector[0])
        print('Size of Encrypted-item-vector', cache_size / (2 ** 20), 'MB')
        communication_time = cache_size * 8 / (band_width * 2 ** 30)
        print('Using a %s Gb/s' % band_width, 'bandwidth, communication will use %s second' % communication_time)

        encrypted_gradient_from_user = []
        user_time_list = []
        test_user_len = 10
        for i in range(test_user_len):
            t = time.time()
            user_vector[i], gradient = user_update(user_vector[i], ratings_dict[user_id_list[i]], encrypted_item_vector)
            user_time_list.append(time.time() - t)
            print('User-%s update using' % i, user_time_list[-1], 'seconds')
            encrypted_gradient_from_user.append(gradient)
        print('User Average time', np.mean(user_time_list))

        # Step 3 Server update
        cache_size = (sys.getsizeof(encrypted_gradient_from_user[0][0][0].ciphertext()) +
                      sys.getsizeof(encrypted_gradient_from_user[0][0][0].exponent)) * \
                     len(encrypted_gradient_from_user[0]) *\
                     len(encrypted_gradient_from_user[0][0])
        print('Size of Encrypted-gradient', cache_size / (2 ** 20), 'MB')
        communication_time = communication_time + cache_size * 8 / (band_width * 2 ** 30)
        print('Using a %s Gb/s' % band_width, 'bandwidth, communication will use %s second' % communication_time)
        t = time.time()
        for g in encrypted_gradient_from_user:
            for i in range(len(encrypted_item_vector)):
                for j in range(len(encrypted_item_vector[i])):
                    encrypted_item_vector[i][j] = encrypted_item_vector[i][j] - g[i][j]
        server_update_time = (time.time() - t) * (len(user_id_list) / test_user_len)
        print('Server update using', server_update_time, 'seconds')

        # for computing loss
        item_vector = np.array([[private_key.decrypt(e) for e in vector] for vector in encrypted_item_vector])
        print('loss', loss())

        print('Costing', max(user_time_list) + server_update_time + communication_time, 'seconds')