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

   from neuro_lit.cli import run_lit
   from neuro_lit.inpaint_image import main as inpaint
   from neuro_lit.data.conform import conform
   from neuro_lit.utils.download_checkpoints import download_models

Most Used Classes
~~~~~~~~~~~~~~~~~

* :class:`neuro_lit.networks.DiffusionUnet.DiffusionUNet`
* :class:`neuro_lit.data.datasets.BrainDataset`
* :class:`neuro_lit.data.transforms.Compose`

Entry Points
~~~~~~~~~~~~

The package provides three main entry points:

1. **lit-download-models**: Download model checkpoints
2. **run-lit**: Main LIT command
3. **lesion-postprocessing**: FastSurfer/FreeSurfer integration pipeline

