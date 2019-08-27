import time
import copy
import numpy as np

from shared_parameter import *
from load_data import train_data, test_data, item_id_list, user_id_list, predict_step

user_vector = np.zeros([len(user_id_list), hidden_dim]) + 0.01

item_vector = np.zeros([len(item_id_list), hidden_dim]) + 0.01


def iterate():
    # User updates
    for i in range(len(user_id_list)):

        for r in range(len(train_data[user_id_list[i]])):

            item_id, rate, _ = train_data[user_id_list[i]][r]

            error = rate - np.dot(user_vector[i], np.transpose(item_vector[item_id]))

            user_vector[i] = user_vector[i] - \
                             lr * (-2 * error * item_vector[item_id] + 2 * reg_u * user_vector[i])

            item_vector[item_id] = item_vector[item_id] - \
                                   lr * (-2 * error * user_vector[i] + 2 * reg_v * item_vector[item_id])


def loss():
    loss = []
    for i in range(len(user_id_list)):
        for r in range(len(train_data[user_id_list[i]])):
            item_id, rate, _ = train_data[user_id_list[i]][r]
            error = (rate - np.dot(user_vector[i], item_vector[item_id])) ** 2
            loss.append(error)
    return np.mean(loss)


if __name__ == '__main__':

    for iteration in range(max_iteration):

        print('#################################')
        print('Iteration', iteration)

        tmp_user_vector = copy.deepcopy(user_vector)
        tmp_item_vector = copy.deepcopy(item_vector)

        t = time.time()
        iterate()
        print('Time', time.time() - t, 's')

        print('loss', loss())

        if np.mean(np.abs(user_vector - tmp_user_vector)) < 1e-4 and\
           np.mean(np.abs(item_vector - tmp_item_vector)) < 1e-4:
            print('Converged')
            break

    np.save('user-%s' % len(user_id_list), user_vector)
    np.save('item-%s' % len(item_id_list), item_vector)

    prediction = []
    real_label = []

    # testing
    for i in range(len(user_id_list)):

        p = np.dot(user_vector[i:i+1], np.transpose(item_vector))[0]

        r = test_data[user_id_list[i]]

        real_label.append([e[1] for e in r])
        prediction.append([p[e[0]] for e in r])

    prediction = np.array(prediction, dtype=np.float32)
    real_label = np.array(real_label, dtype=np.float32)

    print('rmse', np.sqrt(np.mean(np.square(real_label - prediction))))