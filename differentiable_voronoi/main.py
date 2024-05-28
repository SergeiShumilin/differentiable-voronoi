import matplotlib.pyplot as plt
from differentiable_voronoi import differentiable_voronoi, triangulate
import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy


def train(point_cloud, boundary, clamp_min=0, clamp_max=1., n_epochs=100):
    coords = torch.tensor(point_cloud, requires_grad=True)
    optimizer = torch.optim.Adam([coords], lr=0.001)
    mesh = triangulate(coords.clone().detach().numpy())
    loss_curve = []

    for i in tqdm(range(1, n_epochs)):
        optimizer.zero_grad()
        edge_index, areas, e, clipped_vertices_dict = differentiable_voronoi(coords, mesh, boundary, vizualize=False)

        loss = torch.mean((areas - torch.mean(areas)) ** 2)
        loss_curve.append(loss.item())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            coords[:] = coords.clamp(clamp_min, clamp_max)
            mesh = triangulate(coords.clone().detach().numpy())

    return coords, loss_curve


l = 1
eps = 0.0001
coords = np.random.rand(100, 2)
boundary = torch.tensor([[-eps, -eps],
                               [-eps, l + eps],
                               [l + eps, l + eps],
                               [l + eps, -eps]], requires_grad=False)

differentiable_voronoi(torch.tensor(coords, dtype=torch.float),
                       triangulate(coords),
                       vizualize=True,
                       boundary=boundary)

optimized_coords, loss_curve = train(deepcopy(coords), boundary)

plt.plot(loss_curve)

differentiable_voronoi(optimized_coords.detach(),
                       triangulate(optimized_coords.detach().numpy()),
                       vizualize=True,
                       boundary=boundary)