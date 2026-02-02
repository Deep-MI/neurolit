Utils Module
============

.. automodule:: neuro_lit.utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The utils module provides utility functions for various tasks including model downloading, plotting, and general helpers.

Submodules
----------

download_checkpoints
~~~~~~~~~~~~~~~~~~~~

.. automodule:: neuro_lit.utils.download_checkpoints
   :members:
   :undoc-members:
   :show-inheritance:

Download and manage model checkpoints.

**Key Functions:**

- ``download_models()``: Download all model checkpoints
- ``get_model_path()``: Get path to specific model
- ``verify_checksum()``: Verify model integrity

plotting
~~~~~~~~

.. automodule:: neuro_lit.utils.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Visualization utilities for brain MRI and results.

**Key Functions:**

- ``plot_slices()``: Plot 2D slices
- ``plot_comparison()``: Compare before/after inpainting
- ``plot_mask_overlay()``: Overlay mask on image

Examples
--------

Downloading Models
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.utils.download_checkpoints import download_models
   
   # Download all models
   download_models(force=False)  # Skip if already downloaded

Command-Line Download
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Download models
   lit-download-models
   
   # Force re-download
   lit-download-models --force

Getting Model Paths
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.utils.download_checkpoints import get_model_path
   
   # Get path to specific model
   axial_path = get_model_path('axial')
   coronal_path = get_model_path('coronal')
   sagittal_path = get_model_path('sagittal')
   
   print(f"Axial model: {axial_path}")

Plotting Results
~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.utils.plotting import plot_comparison
   import nibabel as nib
   
   # Load images
   original = nib.load('T1w.nii.gz').get_fdata()
   inpainted = nib.load('T1w_inpainted.nii.gz').get_fdata()
   mask = nib.load('lesion_mask.nii.gz').get_fdata()
   
   # Plot comparison
   plot_comparison(
       original=original,
       inpainted=inpainted,
       mask=mask,
       slice_idx=128,
       save_path='comparison.png'
   )

Plotting Slices
~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.utils.plotting import plot_slices
   import nibabel as nib
   
   # Load image
   image = nib.load('T1w.nii.gz').get_fdata()
   
   # Plot axial, coronal, and sagittal slices
   plot_slices(
       image,
       title='T1-weighted MRI',
       save_path='slices.png'
   )

Mask Overlay
~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.utils.plotting import plot_mask_overlay
   import nibabel as nib
   
   # Load data
   image = nib.load('T1w.nii.gz').get_fdata()
   mask = nib.load('lesion_mask.nii.gz').get_fdata()
   
   # Plot with overlay
   plot_mask_overlay(
       image=image,
       mask=mask,
       slice_idx=128,
       alpha=0.5,  # Overlay transparency
       save_path='overlay.png'
   )

Custom Visualization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.utils.plotting import plot_slices
   import matplotlib.pyplot as plt
   import nibabel as nib
   
   # Load multiple images
   original = nib.load('T1w.nii.gz').get_fdata()
   inpainted = nib.load('T1w_inpainted.nii.gz').get_fdata()
   
   # Create custom figure
   fig, axes = plt.subplots(1, 2, figsize=(12, 6))
   
   slice_idx = 128
   axes[0].imshow(original[:, :, slice_idx], cmap='gray')
   axes[0].set_title('Original')
   axes[0].axis('off')
   
   axes[1].imshow(inpainted[:, :, slice_idx], cmap='gray')
   axes[1].set_title('Inpainted')
   axes[1].axis('off')
   
   plt.tight_layout()
   plt.savefig('custom_comparison.png', dpi=300)
   plt.close()

Batch Visualization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.utils.plotting import plot_comparison
   from pathlib import Path
   import nibabel as nib
   
   subjects = ['sub-01', 'sub-02', 'sub-03']
   
   for subject in subjects:
       original = nib.load(f'data/{subject}/T1w.nii.gz').get_fdata()
       inpainted = nib.load(f'output/{subject}/T1w_inpainted.nii.gz').get_fdata()
       mask = nib.load(f'data/{subject}/lesion_mask.nii.gz').get_fdata()
       
       plot_comparison(
           original=original,
           inpainted=inpainted,
           mask=mask,
           slice_idx=128,
           save_path=f'visualizations/{subject}_comparison.png'
       )

