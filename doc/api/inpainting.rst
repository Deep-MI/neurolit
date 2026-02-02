Inpainting Module
=================

.. automodule:: neuro_lit.inpaint_image
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The inpainting module provides direct access to the core inpainting functionality without the full pipeline wrapper.

Main Function
-------------

main
~~~~

:py:func:`neuro_lit.inpaint_image.main`

Core inpainting functionality.

**Usage:**

.. code-block:: bash

   python3 -m neuro_lit.inpaint_image --input_image T1w.nii.gz \\
                 --mask_image mask.nii.gz \\
                 --out_dir output

Examples
--------

Basic Inpainting
~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.inpaint_image import main
   import argparse
   
   args = argparse.Namespace(
       input_image='T1w.nii.gz',
       mask_image='lesion_mask.nii.gz',
       out_dir='output',
       device='cuda',
       batch_size=16,
       num_samples=100
   )
   
   main(args)

Custom Parameters
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.inpaint_image import main
   import argparse
   
   # Use custom parameters
   args = argparse.Namespace(
       input_image='T1w.nii.gz',
       mask_image='lesion_mask.nii.gz',
       out_dir='output',
       device='cuda',
       batch_size=8,  # Smaller batch size
       num_samples=200,  # More diffusion steps
       model_axial='custom_models/axial.pt',
       model_coronal='custom_models/coronal.pt',
       model_sagittal='custom_models/sagittal.pt'
   )
   
   main(args)

CPU Mode
~~~~~~~~

For systems without GPU:

.. code-block:: python

   args = argparse.Namespace(
       input_image='T1w.nii.gz',
       mask_image='lesion_mask.nii.gz',
       out_dir='output',
       device='cpu',  # Use CPU
       batch_size=4,  # Smaller batch for CPU
       num_samples=100
   )
   
   main(args)

Integration with Other Tools
-----------------------------

Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.inpaint_image import main
   from neuro_lit.data.conform import conform_image
   import argparse
   
   # Step 1: Conform image
   conform_image(
       input_path='raw_T1w.nii.gz',
       output_path='T1w_conformed.nii.gz'
   )
   
   # Step 2: Inpaint
   args = argparse.Namespace(
       input_image='T1w_conformed.nii.gz',
       mask_image='lesion_mask.nii.gz',
       out_dir='output',
       device='cuda'
   )
   
   main(args)

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   import subprocess
   from pathlib import Path
   
   data_dir = Path('data')
   output_dir = Path('output')
   
   for subject_dir in data_dir.glob('sub-*'):
       subject_id = subject_dir.name
       
       cmd = [
           'python3', '-m', 'neuro_lit.inpaint_image',
           '--input_image', str(subject_dir / 'T1w.nii.gz'),
           '--mask_image', str(subject_dir / 'lesion_mask.nii.gz'),
           '--out_dir', str(output_dir / subject_id)
       ]
       
       subprocess.run(cmd, check=True)
       print(f"Completed {subject_id}")

