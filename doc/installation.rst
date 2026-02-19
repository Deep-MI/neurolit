Installation
============

neuroLIT can be installed in multiple ways depending on your needs and environment.

Using Docker/Singularity (Recommended)
---------------------------------------

The easiest way to use neuroLIT is through containerization, which handles all dependencies automatically.

**Prerequisites:**

* Docker or Singularity/Apptainer installed on your system

**Quick Start:**

.. code-block:: bash

   git clone https://github.com/Deep-MI/neurolit.git
   cd neurolit
   ./neurolit/scripts/run_lit_containerized.sh --input_image T1w.nii.gz \\
       --mask_image lesion_mask.nii.gz \\
       --output_directory output_directory

By default, this uses Docker. To use Singularity instead:

.. code-block:: bash

   ./neurolit/scripts/run_lit_containerized.sh --singularity \\
       --input_image T1w.nii.gz \\
       --mask_image lesion_mask.nii.gz \\
       --output_directory output_directory

Using PyPI
-----------------

neuroLIT is available on PyPI.

**Installation:**

.. code-block:: bash

   # Install the package
   pip install neurolit

**Download Model Checkpoints:**

After installation, download the required model checkpoints (~500MB):

.. code-block:: bash

   lit-download-models

This will download models to a platform-specific location:

* **Linux:** ``~/.local/share/LIT/weights``
* **macOS:** ``~/Library/Application Support/LIT/weights``
* **Windows:** ``C:\\Users\\<user>\\AppData\\Local\\Deep-MI\\LIT\\weights``

.. note::
   If you skip the ``lit-download-models`` step, models will be automatically downloaded on first use.

From Source (Development Version)
---------------------------------

For development or if you want to modify the code:

**Installation Steps:**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Deep-MI/neurolit.git
   cd neurolit
   
   # Install the package and its dependencies in development mode
   pip install -e .
   
   # Download model checkpoints
   lit-download-models

System Requirements
-------------------

**Minimum Requirements:**

* **RAM:** 8 GB (16 GB recommended)
* **GPU:** NVIDIA GPU with CUDA support (recommended for faster processing)
* **Disk Space:** ~2 GB for models and software
* **OS:** Linux, macOS, or Windows

**Recommended Requirements:**

* **RAM:** 16 GB or more
* **GPU:** NVIDIA GPU with 8 GB VRAM or more
* **CUDA:** Version 11.7 or later
* **OS:** Linux (Ubuntu 20.04 or later)

Verifying Installation
----------------------

After installation, verify that neuroLIT is working correctly:

.. code-block:: bash

   # Check installed version
   lit-inpainting --help
   
   # Check if models are downloaded
   ls ~/.local/share/LIT/weights/  # On Linux

You should see three model files:

* ``model_axial.pt``
* ``model_coronal.pt``
* ``model_sagittal.pt``

Troubleshooting
---------------

Models Not Found
~~~~~~~~~~~~~~~~

If you get a "models not found" error:

1. Manually run: ``lit-download-models``
2. Check that the weights directory exists and contains the three model files
3. Ensure you have write permissions to the data directory

CUDA/GPU Issues
~~~~~~~~~~~~~~~

If PyTorch doesn't detect your GPU:

1. Check that CUDA is installed: ``nvidia-smi``
2. Verify PyTorch CUDA support: ``python -c "import torch; print(torch.cuda.is_available())"``
3. Install the correct PyTorch version for your CUDA version from `pytorch.org <https://pytorch.org/get-started/locally/>`_
