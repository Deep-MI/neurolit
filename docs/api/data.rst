Data Module
===========

.. automodule:: LIT.data
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The data module handles data loading, preprocessing, and transformations.

Submodules
----------

conform
~~~~~~~

.. automodule:: LIT.data.conform
   :members:
   :undoc-members:
   :show-inheritance:

Image conforming utilities for standardizing brain MRI images.

**Key Functions:**

- ``conform_image()``: Conform image to standard space
- ``check_orientation()``: Verify image orientation
- ``resample_image()``: Resample to target resolution

datasets
~~~~~~~~

.. automodule:: LIT.data.datasets
   :members:
   :undoc-members:
   :show-inheritance:

PyTorch dataset classes for brain MRI data.

**Key Classes:**

- ``BrainDataset``: Dataset for brain MRI images
- ``InpaintingDataset``: Dataset for inpainting tasks

transforms
~~~~~~~~~~

.. automodule:: LIT.data.transforms
   :members:
   :undoc-members:
   :show-inheritance:

Data augmentation and transformation utilities.

**Key Classes:**

- ``Compose``: Compose multiple transforms
- ``RandomFlip``: Random horizontal/vertical flip
- ``RandomRotation``: Random rotation
- ``Normalize``: Intensity normalization
- ``ToTensor``: Convert to PyTorch tensor

Examples
--------

Conforming Images
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from LIT.data.conform import conform_image
   
   # Conform a single image
   conform_image(
       input_path='raw_T1w.nii.gz',
       output_path='T1w_conformed.nii.gz',
       target_spacing=(1.0, 1.0, 1.0),
       target_size=(256, 256, 256)
   )

Using Datasets
~~~~~~~~~~~~~~

.. code-block:: python

   from LIT.data.datasets import BrainDataset
   from torch.utils.data import DataLoader
   
   # Create dataset
   dataset = BrainDataset(
       data_dir='training_data',
       transform=None
   )
   
   # Create data loader
   loader = DataLoader(
       dataset,
       batch_size=16,
       shuffle=True,
       num_workers=4
   )
   
   # Iterate over batches
   for batch in loader:
       images = batch['image']
       # Process batch...

Applying Transforms
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from LIT.data.transforms import Compose, RandomFlip, Normalize, ToTensor
   
   # Define transform pipeline
   transform = Compose([
       RandomFlip(p=0.5),
       Normalize(mean=0.5, std=0.5),
       ToTensor()
   ])
   
   # Apply to dataset
   dataset = BrainDataset(
       data_dir='training_data',
       transform=transform
   )

Custom Transforms
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from LIT.data.transforms import Compose
   
   class CustomTransform:
       def __call__(self, image):
           # Your custom transformation
           return modified_image
   
   # Use in pipeline
   transform = Compose([
       CustomTransform(),
       Normalize(),
       ToTensor()
   ])

Batch Conforming
~~~~~~~~~~~~~~~~

.. code-block:: python

   from LIT.data.conform import conform_image
   from pathlib import Path
   
   input_dir = Path('raw_data')
   output_dir = Path('conformed_data')
   output_dir.mkdir(exist_ok=True)
   
   for img_path in input_dir.glob('*.nii.gz'):
       output_path = output_dir / img_path.name
       conform_image(
           input_path=str(img_path),
           output_path=str(output_path)
       )
       print(f"Conformed {img_path.name}")

