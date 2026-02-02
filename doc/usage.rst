Usage Guide
===========

This guide covers the basic and advanced usage of neuro_lit.

Basic Usage
-----------

Running LIT with Containerization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most straightforward way to run LIT is using the containerized wrapper script:

.. code-block:: bash

   ./LIT/scripts/run_lit_containerized.sh \\
       --input_image T1w.nii.gz \\
       --mask_image lesion_mask.nii.gz \\
       --output_directory output_directory \\
       --dilate 2

**Key Parameters:**

* ``--input_image``: Path to the T1-weighted MRI image
* ``--mask_image``: Path to the lesion mask (binary or multi-class)
* ``--output_directory``: Directory where outputs will be saved
* ``--dilate``: Number of times to dilate the lesion mask (default: 0)

Running LIT from PyPI
~~~~~~~~~~~~~~~~~~~~~

If you installed via pip:

.. code-block:: bash

   run-lit \\
       --input_image T1w.nii.gz \\
       --mask_image lesion_mask.nii.gz \\
       --output_directory output_directory \\
       --dilate 2

Mask Dilation
~~~~~~~~~~~~~

We recommend performing mask dilation to account for potential undersegmentation:

.. code-block:: bash

   run-lit --input_image T1w.nii.gz \\
          --mask_image lesion_mask.nii.gz \\
          --output_directory output \\
          --dilate 2  # Dilate mask by 2 voxels

**When to use dilation:**

* **Undersegmentation:** Increase dilation
* **Uncertain boundaries:** Use moderate dilation

Understanding the Outputs
--------------------------

LIT produces several output files in the ``inpainting_volumes`` subdirectory:

Output Files
~~~~~~~~~~~~

* **inpainting_result.nii.gz**: The main output with lesions inpainted.
* **inpainting_mask.nii.gz**: The (dilated) mask used for inpainting in the same space as the input.
* **inpainting_original_image.nii.gz**: The conformed original input image.

File Structure
~~~~~~~~~~~~~~

.. code-block:: text

   output_directory/
   └── inpainting_volumes/
       ├── inpainting_result.nii.gz
       ├── inpainting_mask.nii.gz
       └── inpainting_original_image.nii.gz

.. note::
   If the source image was isotropic, the output images will have the same resolution as the input image. The area outside of the lesion mask is preserved, except for robust rescaling of intensity values.

Advanced Usage
--------------

Direct Inpainting (Python API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For programmatic access, use the direct inpainting function:

.. code-block:: python

   from neuro_lit.inpaint_image import main as inpaint
   import argparse
   
   # Prepare arguments
   args = argparse.Namespace(
       input_image='T1w.nii.gz',
       mask_image='lesion_mask.nii.gz',
       out_dir='output',
       device='cuda',  # or 'cpu'
       batch_size=16,
       num_samples=100
   )
   
   # Run inpainting
   inpaint(args)

Batch Processing
~~~~~~~~~~~~~~~~

For processing multiple subjects, create a simple loop:

.. code-block:: bash

   #!/bin/bash
   
   # List of subjects
   subjects=("sub-01" "sub-02" "sub-03")
   
   for sub in "${subjects[@]}"; do
       echo "Processing $sub..."
       run-lit \\
           --input_image data/${sub}/T1w.nii.gz \\
           --mask_image data/${sub}/lesion_mask.nii.gz \\
           --output_directory output/${sub} \\
           --dilate 2
   done

Or using Python:

.. code-block:: python

   import subprocess
   from pathlib import Path
   
   data_dir = Path("data")
   subjects = ["sub-01", "sub-02", "sub-03"]
   
   for subject in subjects:
       print(f"Processing {subject}...")
       cmd = [
           "run-lit",
           "--input_image", str(data_dir / subject / "T1w.nii.gz"),
           "--mask_image", str(data_dir / subject / "lesion_mask.nii.gz"),
           "--output_directory", f"output/{subject}",
           "--dilate", "2"
       ]
       subprocess.run(cmd, check=True)

Command-Line Interface Reference
---------------------------------

run-lit
~~~~~~~

Main command to run the LIT inpainting.

.. code-block:: text

   run-lit [OPTIONS]

Options:
   --input_image PATH        Path to input T1w image [required]
   --mask_image PATH         Path to lesion mask [required]
   --output_directory PATH   Output directory [required]
   --dilate INTEGER          Number of dilation iterations [default: 0]
   --help                   Show this message and exit

lit-download-models
~~~~~~~~~~~~~~~~~~~

Download required model checkpoints.

.. code-block:: text

   lit-download-models [OPTIONS]

Options:
   --force                  Force re-download even if models exist
   --help                   Show this message and exit

lesion-postprocessing
~~~~~~~~~~~~~~~~~~~~~

Integrate lesion masks into FastSurfer/FreeSurfer outputs.

.. code-block:: text

   lesion-postprocessing [OPTIONS]

Options:
   --subject-id TEXT        Subject ID [required]
   --subjects-dir PATH      Subjects directory [required]
   --skip-segstats          Skip volumetric statistics
   --skip-surface-masking   Skip surface masking
   --help                   Show this message and exit

Best Practices
--------------

Input Data
~~~~~~~~~~

1. **Image Quality:** Use high-quality T1-weighted images (0.8-1 mm isotropic preferred)
2. **Mask Quality:** Ensure lesion masks are accurate; oversegmentation is better than undersegmentation.

Performance
~~~~~~~~~~~

1. **GPU Usage:** Use GPU when available for significant speedup, processing with CPU is not recommended.
2. **Batch Size:** Increase batch size on high-memory GPUs (default: 16)
3. **Mask Size:** Larger lesion masks require longer inference.

Quality Control
~~~~~~~~~~~~~~~

1. **Visual Inspection:** It is recommended to visually inspect inpainting results.
2. **Boundary Check:** Errors can happen especially at the lesion boundaries if the lesion is undersegmented, increasing dilation can help.

Postprocessing
--------------

LIT provides tools to integrate lesion masks into FastSurfer/FreeSurfer segmentation and surface outputs.

Unified Postprocessing Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended way to run postprocessing is using the unified ``lesion-postprocessing`` command. This script handles mapping the lesion mask to multiple segmentation files, running volume statistics (segstats), and performing surface masking.

.. code-block:: bash

   # Setup environment
   export FASTSURFER_HOME=/path/to/FastSurfer
   export FREESURFER_HOME=/path/to/freesurfer
   source $FREESURFER_HOME/SetUpFreeSurfer.sh

   # Run unified postprocessing
   lesion-postprocessing \\
       --subject-id SUBJECT_ID \\
       --subjects-dir /path/to/subjects_dir

**Features:**

* **Installation Validation:** Automatically checks for FastSurfer or FreeSurfer.
* **Dynamic Configuration:** Uses ``segstats_config.json`` for volumetric stats and ``surfstats_config.json`` for surface stats.
* **Surface Stats:** Runs ``mris_anatomical_stats`` calls defined in ``surfstats_config.json``.
* **Surface Masking:** Automatically processes both hemispheres.
* **Anatomy Reports:** Automatically generates reports (Replaced, Reduced, and Adjacent labels) for mappings defined in ``segstats_config.json``.
* **Fine-grained Control:** Flags like ``--skip-segstats`` or ``--skip-surface-masking`` are available.

Individual Postprocessing Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For granular control, you can run individual scripts:

1. **lesion_to_segmentation.py**: Inserts lesion labels into volumetric segmentation and generates anatomy reports.
2. **lesion_to_surface.py**: Projects lesion masks onto cortical surfaces.

Common Issues
-------------

Poor Inpainting Quality
~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Inpainted regions don't look realistic

**Solutions:**

* Increase mask dilation (try 3-5 voxels)
* Check input image quality
* Ensure mask accurately covers the entire lesion
* Verify that the input is a T1-weighted image

Mask Not Applied Correctly
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Output doesn't show inpainting in expected regions

**Solutions:**

* Verify mask and image are in the same space
* Check mask file is binary or has correct labels
* Ensure mask and image have compatible dimensions

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

**Problem:** CUDA out of memory error

**Solutions:**

* Reduce batch size: ``--batch_size 8`` or ``--batch_size 4``
* Use CPU mode (slower): ``--device cpu``
* Process on a machine with more GPU memory

