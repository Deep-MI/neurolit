Inference Module
================

.. automodule:: LIT.inference
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The inference module contains the main pipeline for running LIT inference on brain MRI images.

Key Functions
-------------

run_inference
~~~~~~~~~~~~~

.. autofunction:: LIT.inference.run_inference

Main inference function that orchestrates the entire inpainting pipeline.

load_models
~~~~~~~~~~~

.. autofunction:: LIT.inference.load_models

Loads the three pre-trained models (axial, coronal, sagittal) for multi-view inference.

process_slice
~~~~~~~~~~~~~

.. autofunction:: LIT.inference.process_slice

Processes a single 2D slice through the diffusion model.

Examples
--------

Basic Inference
~~~~~~~~~~~~~~~

.. code-block:: python

   from LIT.inference import run_inference
   import argparse
   
   # Create arguments
   args = argparse.Namespace(
       input_image='T1w.nii.gz',
       mask_image='mask.nii.gz',
       output_dir='output',
       device='cuda',
       num_samples=100
   )
   
   # Run inference
   result = run_inference(args)

Custom Model Loading
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from LIT.inference import load_models
   
   # Load models
   models = load_models(
       model_dir='/path/to/models',
       device='cuda'
   )
   
   axial_model = models['axial']
   coronal_model = models['coronal']
   sagittal_model = models['sagittal']

Advanced Usage
--------------

Multi-View Processing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from LIT.inference import process_slice
   import torch
   
   # Process single slice from multiple views
   slice_data = torch.randn(1, 1, 256, 256)
   mask = torch.zeros(1, 1, 256, 256)
   
   # Axial view
   result_axial = process_slice(
       axial_model,
       slice_data,
       mask,
       num_samples=100
   )
   
   # Repeat for coronal and sagittal...

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from LIT.inference import run_inference
   from pathlib import Path
   
   # Process multiple subjects
   subjects = ['sub-01', 'sub-02', 'sub-03']
   
   for subject in subjects:
       args = argparse.Namespace(
           input_image=f'data/{subject}/T1w.nii.gz',
           mask_image=f'data/{subject}/mask.nii.gz',
           output_dir=f'output/{subject}',
           device='cuda'
       )
       run_inference(args)

