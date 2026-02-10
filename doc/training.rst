Training
========

This guide explains how to train your own neuroLIT models on custom data.

Overview
--------

NeuroLIT uses a Denoising Diffusion Probabilistic Model (DDPM) architecture with a U-Net backbone. The model is trained separately for three orthogonal views (axial, coronal, and sagittal).

Data Preparation
----------------

Conforming Images
~~~~~~~~~~~~~~~~~

All training images must be conformed to a standard space using the provided conform script:

.. code-block:: bash

   python3 neurolit/data/conform.py \\
       --input raw_image.nii.gz \\
       --output conformed_image.nii.gz

The conform script:

* Resamples to 1mm isotropic voxels
* Reorients to standard RAS orientation
* Crops or pads to a consistent size
* Normalizes intensity values

Dataset Structure
~~~~~~~~~~~~~~~~~

Organize your conformed data in a directory structure:

.. code-block:: text

   training_data/
   ├── subject_001/
   │   └── T1w_conformed.nii.gz
   ├── subject_002/
   │   └── T1w_conformed.nii.gz
   └── subject_003/
       └── T1w_conformed.nii.gz

Using Docker for Training
--------------------------

You can use the same Docker image for training. Mount your data directory and run the training script:

.. code-block:: bash

   docker run --gpus all \\
       -v /path/to/training_data:/data \\
       -v /path/to/output:/output \\
       --rm deepmi/lit:latest \\
       python3 /opt/neurolit/train_ddpm.py \\
           --data_dir /data \\
           --output_dir /output \\
           --view axial \\
           --batch_size 16 \\
           --num_epochs 1000
