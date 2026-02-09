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


Quick Reference
---------------

Common Functions
~~~~~~~~~~~~~~~~

.. code-block:: python

   from neurolit.cli import run_lit
   from neurolit.inpaint_image import main as inpaint
   from neurolit.data.conform import conform
   from neurolit.utils.download_checkpoints import download_models

Most Used Classes
~~~~~~~~~~~~~~~~~

* :class:`neurolit.networks.DiffusionUnet.DiffusionUNet`
* :class:`neurolit.data.datasets.BrainDataset`
* :class:`neurolit.data.transforms.Compose`

Entry Points
~~~~~~~~~~~~

The package provides three main entry points:

1. **lit-download-models**: Download model checkpoints
2. **lit-inpainting**: Main LIT command
3. **lit-postprocessing**: FastSurfer/FreeSurfer integration pipeline

