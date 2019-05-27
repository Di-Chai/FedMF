from phe import paillier

hidden_dim = 100

max_iteration = 1

reg_u = 1e-4
reg_v = 1e-4

lr = 1e-3

band_width = 1 # Gb/s

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024, )