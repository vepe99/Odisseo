import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,20))
for i in range(10):
    data = np.load(f'/export/data/vgiusepp/galax_data/data_varying_position_uniform_prior/file_{i:06d}.npz')
    x = data['x']
    theta = data['theta']
    ax = fig.add_subplot(2, 5, i+1, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], )
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel(r'$\alpha$ [rad]')
    ax.set_zlabel(r'$\delta$ [rad]')
    ax.set_title(f'File {i:06d}\n')
plt.savefig('./plot/stream_datasets_pos.png')

fig = plt.figure(figsize=(20,20))
for i in range(10):
    data = np.load(f'/export/data/vgiusepp/galax_data/data_varying_position_uniform_prior/file_{i:06d}.npz')
    x = data['x']
    theta = data['theta']
    ax = fig.add_subplot(2, 5, i+1, projection='3d')
    ax.scatter(x[:, 3], x[:, 4], x[:, 5], )
    ax.set_xlabel(r'$V_R$ ')
    ax.set_ylabel(r'$V_\alpha$')
    ax.set_zlabel(r'$V_\delta$')
    ax.set_title(f'File {i:06d}\n')
plt.savefig('./plot/stream_datasets_vel.png')
