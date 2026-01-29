API Reference
=============

This section contains detailed API documentation for all LIT modules.

.. toctree::
   :maxdepth: 2

   cli
   inference
   inpainting
   data
   networks
   postprocessing
   utils

Module Overview
---------------

LIT Package Structure
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   LIT/
   ├── cli.py                  # Command-line interface
   ├── inference.py            # Main inference pipeline
   ├── inpaint_image.py        # Core inpainting functionality
   ├── train_ddpm.py           # Training script
   ├── data/                   # Data processing modules
   │   ├── conform.py         # Image conforming
   │   ├── datasets.py        # Dataset classes
   │   └── transforms.py      # Data transformations
   ├── networks/              # Neural network architectures
   │   ├── DiffusionUnet.py   # U-Net for diffusion
   │   └── interpolation_layer.py  # Custom layers
   ├── postprocessing/        # Postprocessing tools
   │   ├── lesion_to_segmentation.py  # Mask segmentations
   │   └── lesion_to_surface.py       # Mask surfaces
   └── utils/                 # Utility functions
       ├── download_checkpoints.py  # Model downloading
       ├── plotting.py              # Visualization
       └── logging.py               # Console logging helpers

Quick Reference
---------------

Common Functions
~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.cli import run_lit
   from neuro_lit.inpaint_image import main as inpaint
   from neuro_lit.data.conform import conform
   from neuro_lit.utils.download_checkpoints import download_models

Most Used Classes
~~~~~~~~~~~~~~~~~

* :class:`lit.networks.DiffusionUnet.DiffusionUNet`
* :class:`lit.data.datasets.BrainDataset`
* :class:`lit.data.transforms.Compose`

Entry Points
~~~~~~~~~~~~

The package provides three main entry points:

1. **run-lit**: Full pipeline wrapper
2. **inpaint-image**: Direct inpainting
3. **lit-download-models**: Download model checkpoints

