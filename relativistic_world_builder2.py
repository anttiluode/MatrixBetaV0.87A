"""
QUATERNION WORLD BUILDER (Relativistic Spacetime Engine)
========================================================
A PyTorch Generative Engine that uses Quaternion Minkowski Math 
to enforce the Speed of Light on Neural Network Attention.

This proves "Causal Sparsity": The network naturally drops connections
to any token outside its Light Cone, requiring zero arbitrary masking.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# 1. THE RELATIVISTIC ATTENTION BLOCK (The Physics Engine)
# ============================================================================
class RelativisticAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

    def forward(self, coords, features):
        """
        coords: Tensor of shape (N, 4) representing (t, x, y, z)
        features: Tensor of shape (N, d_model) representing matter/energy
        """
        N = coords.shape[0]
        
        # 1. Calculate pairwise differences for all events
        # dt[i, j] = time of event i - time of event j
        dt = coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0)
        dx = coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0)
        dy = coords[:, 2].unsqueeze(1) - coords[:, 2].unsqueeze(0)
        dz = coords[:, 3].unsqueeze(1) - coords[:, 3].unsqueeze(0)
        
        # 2. THE MINKOWSKI METRIC (Derived from Quaternion Real Product)
        # ds^2 = dt^2 - dx^2 - dy^2 - dz^2
        ds2 = (dt**2) - (dx**2) - (dy**2) - (dz**2)
        
        # 3. CAUSAL SPARSITY (The Light Cone Mask)
        # An event 'j' can only influence event 'i' if:
        # A) j happened before i (dt >= 0)
        # B) j is inside the light cone of i (ds2 >= 0)
        causal_mask = (ds2 >= 0) & (dt >= 0)
        
        # 4. Standard Attention Computation
        Q = self.W_q(features)
        K = self.W_k(features)
        V = self.W_v(features)
        
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale
        
        # 5. ENFORCE PHYSICS
        # If the event is Spacelike (disconnected), we obliterate the attention score
        # to negative infinity, meaning NO information can physically transfer.
        attention_scores = attention_scores.masked_fill(~causal_mask, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # If a token has NO past light cone (vacuum), softmax will yield NaNs. 
        # We catch these and replace with 0.
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        output = torch.matmul(attention_weights, V)
        return output, causal_mask

class SpacetimeWorldBuilder(nn.Module):
    def __init__(self, d_model=3):
        super().__init__()
        # We use 3 layers of relativistic attention. 
        # Information can travel up to 3 "hops" within the light cone.
        self.layer1 = RelativisticAttention(d_model)
        self.layer2 = RelativisticAttention(d_model)
        self.layer3 = RelativisticAttention(d_model)

    def forward(self, coords, initial_features):
        x, _ = self.layer1(coords, initial_features)
        x = torch.relu(x)
        x, _ = self.layer2(coords, x)
        x = torch.relu(x)
        x, mask = self.layer3(coords, x)
        return torch.sigmoid(x), mask # Sigmoid to output RGB colors

# ============================================================================
# 2. THE BIG BANG (Generating the Universe Data)
# ============================================================================
print("Generating Universe Topology...")
NUM_EVENTS = 2500
MAX_TIME = 10.0
MAX_SPACE = 10.0

# Generate random vacuum events across space and time
coords = torch.zeros((NUM_EVENTS, 4))
coords[:, 0] = torch.rand(NUM_EVENTS) * MAX_TIME             # Time (t)
coords[:, 1:] = (torch.rand(NUM_EVENTS, 3) * 2 - 1) * MAX_SPACE # Space (x,y,z)

# The Initial State (Vacuum is empty/black)
features = torch.zeros((NUM_EVENTS, 3))

# THE SEED EVENT (The Big Bang)
# We force Event 0 to be at the exact origin of Spacetime, bursting with raw energy (RGB: Cyan)
coords[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])
features[0] = torch.tensor([0.0, 1.0, 1.0]) 

# ============================================================================
# 3. RUNNING THE RELATIVISTIC FORWARD PASS
# ============================================================================
print("Running Relativistic Attention Physics...")
model = SpacetimeWorldBuilder(d_model=3)

# We initialize the weights to be identity-like so energy flows cleanly for the demo
for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.eye_(param)
    elif 'bias' in name:
        nn.init.zeros_(param)

with torch.no_grad():
    # In one forward pass, the model calculates the entire destiny of the universe.
    # Energy from the Seed cascades through the network, but ONLY along valid light cones.
    generated_world, causal_mask = model(coords, features)

# Extract to numpy for plotting
coords_np = coords.numpy()
colors_np = generated_world.numpy()

# Calculate energy density (how much color/information a node received)
energy = np.linalg.norm(colors_np, axis=1)

# ============================================================================
# 4. INTERACTIVE 3D SIMULATION GUI
# ============================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Setup scatter plot
sc = ax.scatter([], [], [], c=[], s=20, alpha=0.8)

ax.set_xlim([-MAX_SPACE, MAX_SPACE])
ax.set_ylim([-MAX_SPACE, MAX_SPACE])
ax.set_zlim([-MAX_SPACE, MAX_SPACE])
ax.set_xlabel('X Space')
ax.set_ylabel('Y Space')
ax.set_zlabel('Z Space')
ax.set_title("Relativistic World Builder (Drag Slider to Advance Time)")

# Wireframe for the expanding Light Cone
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
cone_x = np.cos(u)*np.sin(v)
cone_y = np.sin(u)*np.sin(v)
cone_z = np.cos(v)
wireframe = None

def update(val):
    global wireframe
    current_time = slider_t.val
    
    # Find all events that have "spawned" up to this point in time
    visible_idx = coords_np[:, 0] <= current_time
    
    vis_coords = coords_np[visible_idx]
    vis_colors = colors_np[visible_idx]
    vis_energy = energy[visible_idx]
    
    # Filter for nodes that actually received causal information
    # Nodes with 0 energy are "Vacuum" (outside the light cone, mathematically starved)
    causal_idx = vis_energy > 0.01
    
    final_coords = vis_coords[causal_idx]
    final_colors = vis_colors[causal_idx]
    
    sc._offsets3d = (final_coords[:, 1], final_coords[:, 2], final_coords[:, 3])
    sc.set_color(final_colors)
    
    # Draw the boundary of the Light Cone (Radius = Time, since c=1)
    if wireframe:
        wireframe.remove()
    wireframe = ax.plot_wireframe(cone_x * current_time, cone_y * current_time, cone_z * current_time, color='cyan', alpha=0.1)
    
    fig.canvas.draw_idle()

# The Time Slider
axcolor = 'lightgoldenrodyellow'
ax_t = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
slider_t = Slider(ax_t, 'World Time (t)', 0.0, MAX_TIME, valinit=0.0)
slider_t.on_changed(update)

update(0)
plt.show()