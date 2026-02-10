Command-Line Interface
======================

.. automodule:: neurolit.cli
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The CLI module provides command-line interfaces for running LIT operations.

Main Functions
--------------

lit-inpainting
~~~~~~~~~~~~~~

:py:func:`neurolit.cli.run_lit`

Entry point for the main ``lit-inpainting`` command. This is the primary interface for running the full LIT pipeline.

**Usage:**

.. code-block:: bash

   lit-inpainting --input_image T1w.nii.gz \\
          --mask_image lesion_mask.nii.gz \\
          --output_directory output

**Parameters:**

- ``--input_image``: Path to input T1-weighted image
- ``--mask_image``: Path to lesion mask
- ``--output_directory``: Output directory path
- ``--dilate``: Number of dilation iterations (default: 0)
- ``--device``: Device to use (cuda/cpu)

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from neurolit.cli import run_lit
   import sys
   
   # Simulate command-line arguments
   sys.argv = [
       'run_lit',
       '--input_image', 'T1w.nii.gz',
       '--mask_image', 'mask.nii.gz',
       '--output_directory', 'output'
   ]
   
   run_lit()

Programmatic Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import subprocess
   
   cmd = [
       'run_lit',
       '--input_image', 'T1w.nii.gz',
       '--mask_image', 'mask.nii.gz',
       '--output_directory', 'output',
       '--dilate', '2'
   ]
   
   result = subprocess.run(cmd, capture_output=True, text=True)
   print(result.stdout)

