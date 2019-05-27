import os
import csv
import numpy as np


def load_csv(fileName, fileWithHeader=True):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        if fileWithHeader:
            header = next(reader)
        else:
            header = []
        data = [r for r in reader]
    return header, data


num_items = 1280 * 2
# num_users = 500

current_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(current_path, 'ml-latest-small')

headers, ratings = load_csv(os.path.join(data_path, 'ratings.csv'))

item_frequent_dict = {}
for e in ratings:
    item_frequent_dict[e[1]] = item_frequent_dict.get(e[1], 0) + 1
item_frequent_dict = sorted(item_frequent_dict.items(), key=lambda x:x[1], reverse=True)


item_id_list = [int(e[0]) for e in item_frequent_dict[:num_items]]
user_id_list = sorted(set([e[0] for e in ratings]), key=lambda x:int(x)) #[:num_users]

print('Number of items', len(item_id_list))
print('Number of users', len(user_id_list))

ratings_dict = {e:[] for e in user_id_list}
counter = 0
for record in ratings:
    if record[0] not in user_id_list or int(record[1]) not in item_id_list:
        continue
    counter += 1
    ratings_dict[record[0]].append([item_id_list.index(int(record[1])), float(record[2])])

print('Number of ratings', counter)