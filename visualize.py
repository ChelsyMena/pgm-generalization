import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def plot_mnist_samples(generated_flat, epoch=0, save_path=None):
    """
    Plots a grid of generated MNIST images.
    generated_flat: (B, 784) numpy array
    """
    # Reshape to (B, 1, 28, 28)
    tensor_data = torch.tensor(generated_flat).view(-1, 1, 28, 28)
    
    # Denormalize [-1, 1] -> [0, 1] for plotting
    tensor_data = (tensor_data + 1) / 2.0
    tensor_data = torch.clamp(tensor_data, 0, 1)
    
    # Create grid
    grid_img = make_grid(tensor_data, nrow=8, padding=2, normalize=False)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f"Generated Samples - Epoch {epoch}")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def solve_trajectory(model, x0, steps=100, device='cpu'):
    """
    Solves the ODE dx/dt = v(x,t) to generate samples.
    """
    model.eval()
    xt = x0.clone().to(device)
    dt = 1.0 / steps
    traj = [xt.cpu().numpy()]
    
    with torch.no_grad():
        for i in range(steps):
            t = torch.ones((x0.shape[0], 1)).to(device) * (i / steps)
            velocity = model(xt, t)
            xt = xt + velocity * dt
            # traj.append(xt.cpu().numpy()) # Uncomment to save full trajectory
    
    traj.append(xt.cpu().numpy())
    return np.array(traj)