import time

from phe import paillier

public_key, private_key = paillier.generate_paillier_keypair(n_length=512)

secret_number_list = [1.14159 for _ in range(5000*100)]

t = time.time()
encrypted_number_list = [public_key.encrypt(x) for x in secret_number_list]
print(time.time() - t)

t = time.time()
print([private_key.decrypt(x) for x in encrypted_number_list])
print(time.time() - t)