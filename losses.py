import torch
import torch.nn.functional as F

def cfm_loss(model, x1, x0, t):
    """
    Standard Conditional Flow Matching Loss.
    """
    # 1. Compute current location x_t
    xt = (1 - t) * x0 + t * x1
    
    # 2. Conditional vector field u_cond (Target velocity)
    u_cond = x1 - x0
    
    # 3. Predict velocity
    v_theta = model(xt, t)
    
    # 4. MSE Loss
    loss = F.mse_loss(v_theta, u_cond)
    return loss