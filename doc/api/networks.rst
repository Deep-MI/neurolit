Networks Module
===============

.. automodule:: lit.networks
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The networks module contains neural network architectures used in lit.

Submodules
----------

DiffusionUnet
~~~~~~~~~~~~~

.. automodule:: lit.networks.DiffusionUnet
   :members:
   :undoc-members:
   :show-inheritance:

U-Net architecture for diffusion-based inpainting.

**Key Classes:**

- ``DiffusionUNet``: Main U-Net model
- ``TimeEmbedding``: Time embedding for diffusion steps
- ``ResidualBlock``: Residual block with time conditioning

interpolation_layer
~~~~~~~~~~~~~~~~~~~

.. automodule:: lit.networks.interpolation_layer
   :members:
   :undoc-members:
   :show-inheritance:

Custom interpolation layers for spatial processing.

**Key Classes:**

- ``InterpolationLayer``: Bilinear/trilinear interpolation
- ``AdaptivePooling``: Adaptive pooling layer

Architecture Details
--------------------

DiffusionUNet Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~

The DiffusionUNet is a modified U-Net with:

- **Encoder**: Downsampling path with residual blocks
- **Bottleneck**: Deepest layer with attention
- **Decoder**: Upsampling path with skip connections
- **Time Conditioning**: Timestep embeddings at each level

.. code-block:: text

   Input (1, 256, 256)
        ↓
   [Encoder Block 1] → [Skip Connection] ──┐
        ↓                                   ↓
   [Encoder Block 2] → [Skip Connection] ──┼─┐
        ↓                                   ↓ ↓
   [Encoder Block 3] → [Skip Connection] ──┼─┼─┐
        ↓                                   ↓ ↓ ↓
   [Bottleneck]                             ↓ ↓ ↓
        ↓                                   ↓ ↓ ↓
   [Decoder Block 3] ←─────────────────────┘ ↓ ↓
        ↓                                     ↓ ↓
   [Decoder Block 2] ←───────────────────────┘ ↓
        ↓                                       ↓
   [Decoder Block 1] ←─────────────────────────┘
        ↓
   Output (1, 256, 256)

Examples
--------

Using DiffusionUNet
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lit.networks.DiffusionUnet import DiffusionUNet
   import torch
   
   # Create model
   model = DiffusionUNet(
       in_channels=1,
       out_channels=1,
       base_channels=64,
       num_res_blocks=2,
       attention_levels=[2, 3]
   )
   
   # Move to GPU
   model = model.cuda()
   model.eval()
   
   # Forward pass
   x = torch.randn(1, 1, 256, 256).cuda()
   t = torch.tensor([100]).cuda()  # Timestep
   
   with torch.no_grad():
       output = model(x, t)

Loading Pre-trained Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lit.networks.DiffusionUnet import DiffusionUNet
   import torch
   
   # Create model
   model = DiffusionUNet()
   
   # Load checkpoint
   checkpoint = torch.load('model_axial.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()

Custom Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lit.networks.DiffusionUnet import DiffusionUNet
   
   # Create custom model
   model = DiffusionUNet(
       in_channels=1,
       out_channels=1,
       base_channels=128,  # More channels
       channel_multipliers=[1, 2, 4, 8],
       num_res_blocks=3,  # More residual blocks
       attention_levels=[2, 3, 4],  # More attention
       dropout=0.1
   )

Training Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   from lit.networks.DiffusionUnet import DiffusionUNet
   import torch
   import torch.nn as nn
   import torch.optim as optim
   
   # Setup
   model = DiffusionUNet().cuda()
   optimizer = optim.Adam(model.parameters(), lr=1e-4)
   criterion = nn.MSELoss()
   
   # Training loop
   model.train()
   for batch in dataloader:
       x = batch['image'].cuda()
       t = batch['timestep'].cuda()
       noise = batch['noise'].cuda()
       
       # Forward pass
       pred_noise = model(x, t)
       loss = criterion(pred_noise, noise)
       
       # Backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

Model Summary
~~~~~~~~~~~~~

.. code-block:: python

   from lit.networks.DiffusionUnet import DiffusionUNet
   
   model = DiffusionUNet()
   
   # Count parameters
   num_params = sum(p.numel() for p in model.parameters())
   print(f"Total parameters: {num_params:,}")
   
   # Count trainable parameters
   num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
   print(f"Trainable parameters: {num_trainable:,}")

