FastSurfer Integration
======================

LIT can be used in conjunction with FastSurfer for whole brain segmentation and cortical surface reconstruction. This guide explains how to integrate LIT with FastSurfer.

Overview
--------

The typical workflow is:

1. **Inpaint lesions** using LIT
2. **Run FastSurfer** on the inpainted image
3. **Mask FastSurfer outputs** to mark lesion regions

This ensures that FastSurfer processes a clean brain image while still marking the lesion locations in the final outputs.

Step 1: Run LIT
---------------

First, inpaint the lesions in your T1-weighted image:

.. code-block:: bash

   run-lit \\
       --input_image T1w.nii.gz \\
       --mask_image lesion_mask.nii.gz \\
       --output_directory lit_output \\
       --dilate 2

This produces an inpainted image at:

.. code-block:: text

   lit_output/inpainting_volumes/T1w_inpainted.nii.gz

Step 2: Run FastSurfer
----------------------

Run FastSurfer on the inpainted image. See the `FastSurfer repository <https://github.com/deep-mi/FastSurfer>`_ for detailed instructions.

Example using Docker:

.. code-block:: bash

   docker run --gpus all \\
       -v /path/to/data:/data \\
       -v /path/to/output:/output \\
       -v /path/to/freesurfer/license:/fs_license \\
       --rm deepmi/fastsurfer:latest \\
       --t1 /data/lit_output/inpainting_volumes/T1w_inpainted.nii.gz \\
       --sid subject_01 \\
       --sd /output \\
       --fs_license /fs_license/license.txt

**Useful FastSurfer Flags:**

* ``--seg_only``: Skip cortical surface reconstruction (much faster!)
* ``--threads 2``: Process both hemispheres in parallel
* ``--py python3.10``: Specify Python version

Step 3: Mask FastSurfer Outputs
--------------------------------

After FastSurfer completes, mask the outputs to mark lesion regions.

Masking Segmentation Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``lesion_to_segmentation.py`` script to mark lesions in the segmentation:

.. code-block:: bash

   python3 LIT/postprocessing/lesion_to_segmentation.py \\
       -i fastsurfer_output/mri/aparc+aseg.mgz \\
       -m lit_output/inpainting_volumes/inpainting_mask.nii.gz \\
       -o fastsurfer_output/mri/aparc+aseg+lesion.mgz

This creates a new segmentation file where lesion voxels are labeled with a special lesion label.

**Parameters:**

* ``-i``: Input FastSurfer segmentation (aparc+aseg.mgz)
* ``-m``: Lesion mask from LIT
* ``-o``: Output segmentation with lesions marked

Masking Surface Files
~~~~~~~~~~~~~~~~~~~~~~

Use the ``lesion_to_surface.py`` script to mark lesions on cortical surfaces:

.. code-block:: bash

   hemisphere="lh"  # or "rh"
   
   python3 LIT/postprocessing/lesion_to_surface.py \\
       --inseg lit_output/inpainting_volumes/inpainting_mask.nii.gz \\
       --insurf fastsurfer_output/surf/${hemisphere}.white.preaparc \\
       --incort fastsurfer_output/label/${hemisphere}.cortex.label \\
       --outaparc fastsurfer_output/label/${hemisphere}.lesion.label \\
       --surflut LIT/postprocessing/DKTatlaslookup_lesion.txt \\
       --seglut LIT/postprocessing/hemi.DKTatlaslookup_lesion.txt \\
       --projmm 0 \\
       --radius 0 \\
       --single_label \\
       --to_annot fastsurfer_output/label/${hemisphere}.aparc.DKTatlas.annot

**Key Parameters:**

* ``--inseg``: Lesion mask from LIT
* ``--insurf``: FastSurfer white matter surface
* ``--incort``: FastSurfer cortex label
* ``--outaparc``: Output lesion label file
* ``--surflut``: Surface lookup table (in LIT repository)
* ``--seglut``: Segmentation lookup table (in LIT repository)
* ``--to_annot``: FastSurfer parcellation annotation to update

Repeat for both hemispheres (lh and rh).

Complete Pipeline Script
-------------------------

Here's a complete script that runs the entire pipeline:

.. code-block:: bash

   #!/bin/bash
   
   # Configuration
   SUBJECT="sub-01"
   T1W_IMAGE="data/${SUBJECT}/T1w.nii.gz"
   LESION_MASK="data/${SUBJECT}/lesion_mask.nii.gz"
   LIT_OUTPUT="output/${SUBJECT}/lit"
   FS_OUTPUT="output/${SUBJECT}/fastsurfer"
   FS_LICENSE="freesurfer/license.txt"
   
   echo "=== Step 1: Running LIT ==="
   run-lit \\
       --input_image ${T1W_IMAGE} \\
       --mask_image ${LESION_MASK} \\
       --output_directory ${LIT_OUTPUT} \\
       --dilate 2
   
   echo "=== Step 2: Running FastSurfer ==="
   docker run --gpus all \\
       -v $(pwd):/data \\
       --rm deepmi/fastsurfer:latest \\
       --t1 /data/${LIT_OUTPUT}/inpainting_volumes/T1w_inpainted.nii.gz \\
       --sid ${SUBJECT} \\
       --sd /data/${FS_OUTPUT} \\
       --fs_license /data/${FS_LICENSE}
   
   echo "=== Step 3: Masking Segmentation ==="
   python3 LIT/postprocessing/lesion_to_segmentation.py \\
       -i ${FS_OUTPUT}/${SUBJECT}/mri/aparc+aseg.mgz \\
       -m ${LIT_OUTPUT}/inpainting_volumes/inpainting_mask.nii.gz \\
       -o ${FS_OUTPUT}/${SUBJECT}/mri/aparc+aseg+lesion.mgz
   
   echo "=== Step 4: Masking Surfaces ==="
   for hemi in lh rh; do
       echo "Processing ${hemi}..."
       python3 LIT/postprocessing/lesion_to_surface.py \\
           --inseg ${LIT_OUTPUT}/inpainting_volumes/inpainting_mask.nii.gz \\
           --insurf ${FS_OUTPUT}/${SUBJECT}/surf/${hemi}.white.preaparc \\
           --incort ${FS_OUTPUT}/${SUBJECT}/label/${hemi}.cortex.label \\
           --outaparc ${FS_OUTPUT}/${SUBJECT}/label/${hemi}.lesion.label \\
           --surflut LIT/postprocessing/DKTatlaslookup_lesion.txt \\
           --seglut LIT/postprocessing/hemi.DKTatlaslookup_lesion.txt \\
           --projmm 0 \\
           --radius 0 \\
           --single_label \\
           --to_annot ${FS_OUTPUT}/${SUBJECT}/label/${hemi}.aparc.DKTatlas.annot
   done
   
   echo "=== Pipeline Complete ==="

Output Structure
----------------

After running the complete pipeline, your output directory will contain:

.. code-block:: text

   output/sub-01/
   ├── lit/
   │   └── inpainting_volumes/
   │       ├── T1w_inpainted.nii.gz
   │       ├── inpainting_mask.nii.gz
   │       └── T1w_inpainted_cropped.nii.gz
   └── fastsurfer/
       └── sub-01/
           ├── mri/
           │   ├── aparc+aseg.mgz
           │   ├── aparc+aseg+lesion.mgz  # With lesions marked
           │   └── ...
           ├── surf/
           │   ├── lh.white
           │   ├── rh.white
           │   └── ...
           └── label/
               ├── lh.lesion.label  # Lesion vertices
               ├── rh.lesion.label
               └── ...

Visualization
-------------

You can visualize the results using FreeSurfer's FreeView:

.. code-block:: bash

   # View segmentation with lesions
   freeview -v fastsurfer_output/sub-01/mri/aparc+aseg+lesion.mgz:colormap=lut
   
   # View surfaces with lesion labels
   freeview -f fastsurfer_output/sub-01/surf/lh.white:label=fastsurfer_output/sub-01/label/lh.lesion.label

Advanced Options
----------------

Smooth Parcellation
~~~~~~~~~~~~~~~~~~~

To smooth the parcellation around lesions:

.. code-block:: bash

   python3 LIT/postprocessing/smooth_aparc.py \\
       --input fastsurfer_output/sub-01/label/lh.aparc.DKTatlas.annot \\
       --output fastsurfer_output/sub-01/label/lh.aparc.DKTatlas.smooth.annot \\
       --surface fastsurfer_output/sub-01/surf/lh.white

Customizing Lesion Labels
~~~~~~~~~~~~~~~~~~~~~~~~~~

The lesion label can be customized by modifying the lookup tables:

* ``DKTatlaslookup_lesion.txt``: Surface label lookup
* ``hemi.DKTatlaslookup_lesion.txt``: Hemisphere label lookup

Best Practices
--------------

1. **Always use dilation (2-3 voxels)** when running LIT before FastSurfer
2. **Keep both versions** of segmentation files (with and without lesion labels)
3. **Process both hemispheres** even if lesion is unilateral
4. **Visual QC** of both inpainting and FastSurfer results
5. **Document parameters** used for reproducibility

Troubleshooting
---------------

FastSurfer Fails on Inpainted Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** FastSurfer crashes or produces poor results

**Solutions:**

* Check inpainting quality visually
* Increase LIT mask dilation
* Ensure inpainted image has reasonable intensity values

Lesion Mask Not Aligned
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Lesion appears in wrong location in FastSurfer outputs

**Solutions:**

* Verify that mask and T1w image are in the same space
* Check that LIT preserved the image geometry
* Use the ``inpainting_mask.nii.gz`` from LIT output (not the original mask)

Missing Surface Labels
~~~~~~~~~~~~~~~~~~~~~~

**Problem:** No lesion vertices appear on surfaces

**Solutions:**

* Ensure lesion affects cortical regions
* Check that surface files exist before masking
* Verify lookup table paths are correct
* Try increasing ``--projmm`` parameter (e.g., 2)

