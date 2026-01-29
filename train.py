import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import time
import csv
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from models import TimeMLP  # Ensure this matches your file structure

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2048  # Large batch for speed
LR = 1e-3         
EPOCHS = 10000
PLOT_INTERVAL = 50
PLOT_FOLDER = "mnist_cfm_stable"
os.makedirs(PLOT_FOLDER, exist_ok=True)

# --- 1. PRE-LOAD DATA (The Speed Fix) ---
print("🚀 Pre-loading MNIST to GPU VRAM...")
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False)

# Normalize to [-1, 1] and move to GPU
X_train = train_set.data.float() / 255.0
X_train = (X_train - 0.5) / 0.5
X_train = X_train.view(-1, 784).to(DEVICE)

X_test = test_set.data.float() / 255.0
X_test = (X_test - 0.5) / 0.5
X_test = X_test.view(-1, 784).to(DEVICE)
print(f"✅ Data loaded on {DEVICE}: {X_train.shape}")

# --- MODEL ---
model = TimeMLP(dim=784, hidden_dim=1024, time_embed_dim=128).to(DEVICE)

# No AMP Scaler (Standard FP32 is more stable for this)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(DEVICE)

# --- ROBUST UTILS (Fixes crashes) ---

def compute_lpips_safe(gen_flat, real_flat):
    # Safety: Check for NaNs before they crash LPIPS
    if torch.isnan(gen_flat).any() or torch.isinf(gen_flat).any():
        print("⚠️ Warning: Generated samples contain NaNs. Skipping LPIPS.")
        return 1.0  # Return worst score instead of crashing

    gen = gen_flat.view(-1, 1, 28, 28).clamp(-1, 1)
    real = real_flat.view(-1, 1, 28, 28).clamp(-1, 1)
    
    # Resize to 64x64 for AlexNet
    gen = F.interpolate(gen, size=(64, 64), mode='bilinear').repeat(1, 3, 1, 1)
    real = F.interpolate(real, size=(64, 64), mode='bilinear').repeat(1, 3, 1, 1)
    
    with torch.no_grad():
        return lpips(gen, real).item()

def generate_samples(model, n_samples=64, steps=50):
    model.eval()
    x = torch.randn(n_samples, 784).to(DEVICE)
    dt = 1.0 / steps
    
    with torch.no_grad():
        for i in range(steps):
            t_val = i / steps
            t = torch.ones(n_samples).to(DEVICE) * t_val
            
            # Heun's Method Step 1
            v_1 = model(x, t)
            x_next = x + v_1 * dt
            
            # Heun's Method Step 2
            t_next = (i + 1) / steps
            t_n = torch.ones(n_samples).to(DEVICE) * t_next
            v_2 = model(x_next, t_n)
            
            x = x + 0.5 * (v_1 + v_2) * dt
            
            # Safety Clamp: Prevents numerical explosion during sampling
            x = x.clamp(-3.0, 3.0)
            
    return x

# --- TRAINING LOOP ---

print(f"🔥 Starting Robust Training...")
log_file = os.path.join(PLOT_FOLDER, "log.csv")
with open(log_file, "w", newline='') as f:
    csv.writer(f).writerow(["Epoch", "Loss", "LPIPS", "Time"])

N = X_train.shape[0]
indices = torch.arange(N, device=DEVICE)

for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()
    
    indices = indices[torch.randperm(N, device=DEVICE)]
    total_loss = 0
    
    for start_idx in range(0, N, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, N)
        batch_idx = indices[start_idx:end_idx]
        x1 = X_train[batch_idx]
        
        # Flow Matching Setup
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=DEVICE)
        t_exp = t.view(-1, 1)
        
        # Interpolate
        xt = (1 - t_exp) * x0 + t_exp * x1 + 1e-4 * torch.randn_like(x1)
        v_target = x1 - x0
        
        # Forward Pass
        v_pred = model(xt, t)
        loss = F.mse_loss(v_pred, v_target)
        
        # Immediate NaN check
        if torch.isnan(loss):
            print(f"❌ Loss is NaN at epoch {epoch}! Reseting optimizer...")
            optimizer.zero_grad() # Flush bad grads
            continue # Skip this batch

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # GRADIENT CLIPPING (Crucial for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.detach()

    torch.cuda.synchronize()
    t1 = time.time()
    
    num_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    avg_loss = total_loss.item() / num_batches
    epoch_duration = t1 - t0

    # --- PERIODIC EVALUATION ---
    if epoch % PLOT_INTERVAL == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch}: Evaluating...")
        gen_samples = generate_samples(model, n_samples=64)
        real_batch = X_test[:64]
        score = compute_lpips_safe(gen_samples, real_batch)
        
        gen_img = ((gen_samples.reshape(-1, 28, 28) + 1) / 2).clip(0, 1)
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(gen_img[i].cpu().numpy(), cmap='gray')
            ax.axis('off')
        plt.savefig(os.path.join(PLOT_FOLDER, f"epoch_{epoch}.png"))
        plt.close()

        # Log to CSV
        with open(log_file, "a", newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, score, epoch_duration])

        print(f"  [EVAL] Loss: {avg_loss:.4f} | LPIPS: {score:.4f} | Time: {epoch_duration:.2f}s")
        plt.close()
        
        with open(log_file, "a", newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, score, epoch_duration])
            
        print(f"  [EVAL] Loss: {avg_loss:.4f} | LPIPS: {score:.4f} | Time: {epoch_duration:.2f}s")
    
    else:
        print(f"Epoch {epoch}: Loss: {avg_loss:.4f} | Time: {epoch_duration:.4f}s")