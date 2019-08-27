import matplotlib.pyplot as plt

with open('FedMF-Full.txt', 'r', encoding='utf-8') as f:
    fedmf_full = f.readlines()
with open('FedMF-Part.txt', 'r', encoding='utf-8') as f:
    fedmf_part = f.readlines()
with open('Regular_MF.txt', 'r', encoding='utf-8') as f:
    mf = f.readlines()

fedmf_full = [e.strip('\n') for e in fedmf_full if e.startswith('loss')]
fedmf_full = [float(e.split(' ')[-1]) for e in fedmf_full]

fedmf_part = [e.strip('\n') for e in fedmf_part if e.startswith('loss')]
fedmf_part = [float(e.split(' ')[-1]) for e in fedmf_part]

mf = [e.strip('\n') for e in mf if e.startswith('loss')]
mf = [float(e.split(' ')[-1]) for e in mf]

fig, axs = plt.subplots()

axs.plot(fedmf_full, 'b+-', label='FedMF-Full', linewidth=0.5)
axs.plot(fedmf_part, 'g.-', label='FedMF-Part', linewidth=0.5)
axs.plot(mf, 'r-', label='Regular-MF', linewidth=0.5)

axs.grid()
axs.legend(fontsize=15)

axs.set_xlabel('Epochs', fontsize=15)
axs.set_ylabel('Train Loss', fontsize=15)

fig.set_size_inches(10, 5)
fig.savefig('%s.png' % 'comparison', dpi=100)
plt.close()