import sys
import time
import numpy as np

from shared_parameter import *
from load_data import train_data, test_data, user_id_list, item_id_list


def user_update(single_user_vector, user_rating_list, encrypted_item_vector):

    item_vector = np.array([[private_key.decrypt(e) for e in vector] for vector in encrypted_item_vector],
                           dtype=np.float32)

    gradient = {}
    for item_id, rate, _ in user_rating_list:
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        single_user_vector = single_user_vector - lr * (-2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = lr * (-2 * error * single_user_vector + 2 * reg_v * item_vector[item_id])

    encrypted_gradient = {vector: [public_key.encrypt(e, precision=1e-5) for e in gradient[vector]] for vector in gradient}

    return single_user_vector, encrypted_gradient


def loss():
    loss = []
    # User updates
    for i in range(len(user_id_list)):
        for r in range(len(train_data[user_id_list[i]])):
            item_id, rate, _ = train_data[user_id_list[i]][r]
            error = (rate - np.dot(user_vector[i], item_vector[item_id])) ** 2
            loss.append(error)
    return np.mean(loss)


if __name__ == '__main__':

    # Init process
    user_vector = np.zeros([len(user_id_list), hidden_dim]) + 0.01

    item_vector = np.zeros([len(item_id_list), hidden_dim]) + 0.01

    # Step 1 Server encrypt item-vector
    t = time.time()
    encrypted_item_vector = [[public_key.encrypt(e, precision=1e-5) for e in vector] for vector in item_vector]
    print('Item profile encrypt using', time.time() - t, 'seconds')

    for iteration in range(max_iteration):

        print('###################')
        print('Iteration', iteration)

        # Step 2 User updates
        cache_size = (sys.getsizeof(encrypted_item_vector[0][0].ciphertext()) +
                      sys.getsizeof(encrypted_item_vector[0][0].exponent)) * \
                      len(encrypted_item_vector) * \
                      len(encrypted_item_vector[0])
        print('Size of Encrypted-item-vector', cache_size / (2 ** 20), 'MB')
        communication_time = cache_size * 8 / (band_width * 2 ** 30)
        print('Using a %s Gb/s' % band_width, 'bandwidth, communication will use %s second' % communication_time)

        encrypted_gradient_from_user = []
        user_time_list = []
        for i in range(len(user_id_list)):
            t = time.time()
            user_vector[i], gradient = user_update(user_vector[i], train_data[user_id_list[i]], encrypted_item_vector)
            user_time_list.append(time.time() - t)
            print('User-%s update using' % i, user_time_list[-1], 'seconds')
            encrypted_gradient_from_user.append(gradient)
        print('User Average time', np.mean(user_time_list))

        # Step 3 Server update
        cache_size = np.mean([np.sum([[sys.getsizeof(e.ciphertext()) + sys.getsizeof(e.exponent)
                                       for e in value] for key, value in g.items()])
                              for g in encrypted_gradient_from_user])
        print('Size of Encrypted-gradient', cache_size / (2 ** 20), 'MB')
        communication_time = communication_time + cache_size * 8 / (band_width * 2 ** 30)
        print('Using a %s Gb/s' % band_width, 'bandwidth, communication will use %s second' % communication_time)
        t = time.time()
        for g in encrypted_gradient_from_user:
            for item_id in g:
                for j in range(len(encrypted_item_vector[item_id])):
                    encrypted_item_vector[item_id][j] = encrypted_item_vector[item_id][j] - g[item_id][j]
        server_update_time = (time.time() - t) * (len(user_id_list) / len(user_id_list))
        print('Server update using', server_update_time, 'seconds')

        # for computing loss
        item_vector = np.array([[private_key.decrypt(e) for e in vector] for vector in encrypted_item_vector])
        print('loss', loss())

        print('Costing', max(user_time_list) + server_update_time + communication_time, 'seconds')


    prediction = []
    real_label = []

    # testing
    for i in range(len(user_id_list)):
        p = np.dot(user_vector[i:i + 1], np.transpose(item_vector))[0]

        r = test_data[user_id_list[i]]

        real_label.append([e[1] for e in r])
        prediction.append([p[e[0]] for e in r])

    prediction = np.array(prediction, dtype=np.float32)
    real_label = np.array(real_label, dtype=np.float32)

    print('rmse', np.sqrt(np.mean(np.square(real_label - prediction))))