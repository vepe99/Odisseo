from autocvd import autocvd
autocvd(num_gpus = 1)

# ignore warnings for readability
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
# import torch_geometric as pyg
import pandas as pd
import seaborn as sns
import tarp

import torch
from torch import nn
import torch.nn.functional as F

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import (
    PlotSinglePosterior, PosteriorSamples, PosteriorCoverage)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import re

from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from odisseo.units import CodeUnits
from astropy import units as u

code_length = 10.0 * u.kpc
code_mass = 1e4 * u.Msun
G = 1 
code_units = CodeUnits(code_length, code_mass, G=G)



params_true = SimulationParams(t_end = (3 * u.Gyr).to(code_units.code_time).value,  
                          Plummer_params= PlummerParams(Mtot=(10**4.05 * u.Msun).to(code_units.code_mass).value,
                                                        a=(8 * u.pc).to(code_units.code_length).value),
                           MN_params= MNParams(M = (68_193_902_782.346756 * u.Msun).to(code_units.code_mass).value,
                                              a = (3.0 * u.kpc).to(code_units.code_length).value,
                                              b = (0.280 * u.kpc).to(code_units.code_length).value),
                          NFW_params= NFWParams(Mvir=(4.3683325e11 * u.Msun).to(code_units.code_mass).value,
                                               r_s= (16.0 * u.kpc).to(code_units.code_length).value,),      
                          PSP_params= PSPParams(M = 4501365375.06545 * u.Msun.to(code_units.code_mass),
                                                alpha = 1.8, 
                                                r_c = (1.9*u.kpc).to(code_units.code_length).value),                    
                          G=code_units.G, ) 


class DiskDataset(Dataset):
    def __init__(self, file_paths, file_type="npy", transform=None, code_units=code_units):
        """
        Args:
            file_paths (List[str]): List of paths to data files.
            file_type (str): Type of file to load ('npy', 'pt', or custom).
            transform (callable, optional): Optional transform to apply.
        """
        self.file_paths = self._filter_valid_paths(file_paths)
        self.file_type = file_type
        self.transform = transform

        path = self.file_paths[0]
        if self.file_type == "npz":
            x = np.load(path)['x'], 
            theta = np.load(path)['theta'] 
            x, theta = torch.from_numpy(x), torch.from_numpy(theta)
            x = self.transform_in_histogram(x)
        self.tensors = x.unsqueeze(0), theta.unsqueeze(0)
        print(self.tensors[0].shape, self.tensors[1].shape)

    def _filter_valid_paths(self, paths):
        valid_paths = []
        for path in paths:
            try:
                data = np.load(path)
                x, theta = data['x'], data['theta']
                if not np.isnan(x).any() and not np.isnan(theta).any():
                    valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
        return valid_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        path = self.file_paths[idx]
        x, theta = np.load(path)['x'], np.load(path)['theta']
        x, theta = torch.from_numpy(x), torch.from_numpy(theta)
        
        x = self.transform_in_histogram(x)

        self.tensors = x, theta

        return x, theta
    
    def transform_in_histogram(self, x, bins=[64, 32]):
        ph1_ph2 = torch.histogramdd(input=torch.tensor(x[:, 1], x[:, 2]), bins = bins, range = [-120., 70., -8, 2] )
        R_vR = torch.histogramdd(input=torch.tensor(x[:, 0], x[:, 3]), bins = bins, range = [6., 20., -250., 250.] )
        vphicosphi2_vphi2 = torch.histogramdd(input=torch.tensor(x[:, 4], x[:, 5]), bins = bins, range = [-2., 1., -0.1, 0.1 ] )
        return torch.stack([ph1_ph2, R_vR, vphicosphi2_vphi2], dim=0)


#load data paths
data_path = '/export/data/vgiusepp/odisseo_data/data/data_NFW/'
pattern = re.compile(r"chunk_(\d+)\.npz")  # capture any number of digits

#Training
files_path_training = sorted(
    f for f in Path(data_path).glob("chunk_*.npz")
    if (m := pattern.fullmatch(f.name)) and int(m.group(1)) < 50_000)
dataset_training = DiskDataset(files_path_training, file_type="npz", )
dataloader_training = DataLoader(dataset_training, batch_size=256, shuffle=True, num_workers=4)

for batch in dataloader_training:
    print(batch[0].shape, batch[1].shape)
    break
# Validation
files_path_validation = sorted(
    f for f in Path(data_path).glob("chunk_*.npz")
    if (m := pattern.fullmatch(f.name)) and int(m.group(1)) < 60_000 and int(m.group(1)) >= 50_000)
dataset_validation = DiskDataset(files_path_validation, file_type="npz", )
dataloader_validation = DataLoader(dataset_validation, batch_size=256, shuffle=False, num_workers=4)
for batch in dataloader_validation:
    print(batch[0].shape, batch[1].shape)
    break
# Test
files_path_test = sorted(
    f for f in Path(data_path).glob("chunk_*.npz")
    if (m := pattern.fullmatch(f.name)) and int(m.group(1)) < 61_000 and int(m.group(1)) >= 60_000)
dataset_test = DiskDataset(files_path_test, file_type="npz", )
dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=4)
for batch in dataloader_test:
    print(batch[0].shape, batch[1].shape)
    break

# Observation
files_path_obs = sorted(
    f for f in Path(data_path).glob("chunk_*.npz")
    if (m := pattern.fullmatch(f.name)) and int(m.group(1)) == 70_000)
x_obs = torch.from_numpy(np.load(files_path_obs[0])['x'])
theta_obs = torch.from_numpy(np.load(files_path_obs[0])['theta'])
print(theta_obs)


class Unet(nn.Module):
    def __init__(self, input_dim, output_dim,):
        pass
    
embedding_net = Unet(hidden_dim=128, output_dim=128)


# ltu-ili 
loader = ili.dataloaders.TorchLoader(
    train_loader=dataloader_training,
    val_loader=dataloader_validation,
)

trainer = ili.inference.InferenceRunner.load(
  backend = 'lampe',          # Choose a backend and inference engine (here, Neural Posterior Estimation)
  engine='NPE',               
  # define a prior
  prior = ili.utils.Uniform(low=[0.5, # t_end in Gyr
                                np.log10(10**3.0).item(), # Plummer mass
                                params_true.Plummer_params.a*(1/4) * code_units.code_length.to(u.kpc),
                                np.log10(params_true.NFW_params.Mvir*(1/4) * code_units.code_mass.to(u.Msun)).item(),
                                params_true.NFW_params.r_s*(1/4) * code_units.code_length.to(u.kpc), 
                                np.log10(params_true.MN_params.M*(1/4) * code_units.code_mass.to(u.Msun)).item(), 
                                params_true.MN_params.a*(1/4) * code_units.code_length.to(u.kpc), ], 
                            high=[5, # t_end in Gyr
                                np.log10(10**4.5).item(), #Plummer mass
                                params_true.Plummer_params.a*(8/4) * code_units.code_length.to(u.kpc),
                                np.log10(params_true.NFW_params.Mvir*(8/4) * code_units.code_mass.to(u.Msun)).item(), 
                                params_true.NFW_params.r_s*(8/4) * code_units.code_length.to(u.kpc), 
                                np.log10(params_true.MN_params.M*(8/4) * code_units.code_mass.to(u.Msun)).item(), 
                                params_true.MN_params.a*(8/4) * code_units.code_length.to(u.kpc),], 
                            device=device),

  # Define a neural network architecture (here, MAF)
  nets = [ili.utils.load_nde_lampe(engine='NPE', 
                                   model='nsf', 
                                   embedding_net=embedding_net, 
                                   x_normalize=True,
                                   theta_normalize=True,),
         ili.utils.load_nde_lampe(engine='NPE', 
                                   model='nsf', 
                                   embedding_net=embedding_net, 
                                   x_normalize=True,
                                   theta_normalize=True,)],
  device = device,
  train_args = {'stop_after_epochs': 5, 'max_epochs': 200}
)

posterior, summary = trainer(loader)                  # Run training to map data -> parameters

# samples = posterior.sample(                     # Generate 1000 samples from the posterior for input x[0]
#   x=x_obs, shape=(1000,)
# )

# # plot train/validation loss
# fig, ax = plt.subplots(1, 1, figsize=(6,4))
# c = list(mcolors.TABLEAU_COLORS)
# for i, m in enumerate(summary):
#     ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i])
#     ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
# ax.set_xlim(0)
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Log probability')
# ax.legend()

# fig.savefig("log_probabilities.pdf", bbox_inches='tight', format='pdf')

# use ltu-ili's built-in validation metrics to plot the posterior for this point
metric = PlotSinglePosterior(
    num_samples=1000, sample_method='direct', 
    labels=[f'$\\theta_{i}$' for i in range(5)]
)
fig = metric(
    posterior=posterior,
    x_obs = x_obs, theta_fid=theta_obs,
)
fig.savefig("posterior_plot.pdf", bbox_inches='tight', format='pdf')

files_path_test = sorted(
    f for f in Path(data_path).glob("chunk_*.npz")
    if (m := pattern.fullmatch(f.name)) and int(m.group(1)) < 61_000 and int(m.group(1)) >= 60_000
)

dataset_test = DiskDataset(files_path_test, file_type="npz", )
dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=4)
for batch in dataloader_test:
    print(batch[0].shape, batch[1].shape)
    break


def dataloader_to_tensors(dataloader):
    xs, thetas = [], []
    for x_batch, theta_batch in dataloader:
        xs.append(x_batch)
        thetas.append(theta_batch)
    x_tensor = torch.cat(xs, dim=0)
    theta_tensor = torch.cat(thetas, dim=0)
    return x_tensor, theta_tensor

x, theta = dataloader_to_tensors(dataloader_test)

# x = torch.tensor([a[0] for a in dataloader_test])
# theta = torch.tensor([a[1] for a in dataloader_test])                 

metric = PosteriorCoverage(
    num_samples=1000, sample_method='direct', 
    labels=[f'$\\theta_{i}$' for i in range(5)],
    plot_list = ["coverage", "histogram", "predictions", "tarp"],
    out_dir=None
)


fig = metric(
    posterior=posterior,
    x=x, theta=theta
)

plot_list = ["coverage", "histogram", "predictions", "tarp"]
for i, f in enumerate(fig):
    name= plot_list[i]
    f.savefig(f"./test_plot/posterior_coverage_{name}.pdf", bbox_inches='tight', format='pdf')
               
metric = PosteriorCoverage(
    num_samples=1000, sample_method='direct', 
    labels=[f'$\\theta_{i}$' for i in range(5)],
    plot_list = ["coverage", "histogram", "predictions", "tarp"],
    out_dir=None
)


fig = metric(
    posterior=posterior,
    x=x, theta=theta
)

plot_list = ["coverage", "histogram", "predictions", "tarp"]
for i, f in enumerate(fig):
    name= plot_list[i]
    f.savefig(f"posterior_coverage_{name}.pdf", bbox_inches='tight', format='pdf')