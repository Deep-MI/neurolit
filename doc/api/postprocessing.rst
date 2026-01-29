Postprocessing Module
=====================

.. automodule:: lit.postprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The postprocessing module provides tools for masking FastSurfer outputs with lesion information.

Submodules
----------

lesion_to_segmentation
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: lit.postprocessing.lesion_to_segmentation
   :members:
   :undoc-members:
   :show-inheritance:

Mask lesions in volumetric segmentation files.

**Key Functions:**

- ``mask_segmentation()``: Apply lesion mask to segmentation
- ``read_segmentation()``: Read segmentation file
- ``write_segmentation()``: Write segmentation file

lesion_to_surface
~~~~~~~~~~~~~~~~~

.. automodule:: lit.postprocessing.lesion_to_surface
   :members:
   :undoc-members:
   :show-inheritance:

Mark lesion vertices on cortical surfaces.

**Key Functions:**

- ``project_mask_to_surface()``: Project lesion mask to surface
- ``find_lesion_vertices()``: Identify affected vertices
- ``create_lesion_label()``: Create label file

Examples
--------

Masking Segmentations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.postprocessing.lesion_to_segmentation import mask_segmentation
   
   # Apply lesion mask to segmentation
   mask_segmentation(
       input_seg='fastsurfer/mri/aparc+aseg.mgz',
       lesion_mask='lit_output/inpainting_mask.nii.gz',
       output_seg='fastsurfer/mri/aparc+aseg+lesion.mgz',
       lesion_label=99  # Label value for lesions
   )

Command-Line Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 LIT/postprocessing/lesion_to_segmentation.py \\
       -i fastsurfer/mri/aparc+aseg.mgz \\
       -m lit_output/inpainting_mask.nii.gz \\
       -o fastsurfer/mri/aparc+aseg+lesion.mgz

Masking Surfaces
~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.postprocessing.lesion_to_surface import project_mask_to_surface
   
   # Project lesion mask to surface
   project_mask_to_surface(
       lesion_mask='lit_output/inpainting_mask.nii.gz',
       surface_file='fastsurfer/surf/lh.white',
       cortex_label='fastsurfer/label/lh.cortex.label',
       output_label='fastsurfer/label/lh.lesion.label',
       projection_mm=0,  # Distance to project
       radius=0  # Search radius
   )

Command-Line Surface Masking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 LIT/postprocessing/lesion_to_surface.py \\
       --inseg lit_output/inpainting_mask.nii.gz \\
       --insurf fastsurfer/surf/lh.white.preaparc \\
       --incort fastsurfer/label/lh.cortex.label \\
       --outaparc fastsurfer/label/lh.lesion.label \\
       --surflut LIT/postprocessing/DKTatlaslookup_lesion.txt \\
       --seglut LIT/postprocessing/hemi.DKTatlaslookup_lesion.txt \\
       --projmm 0 \\
       --radius 0 \\
       --single_label \\
       --to_annot fastsurfer/label/lh.aparc.DKTatlas.annot

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from neuro_lit.postprocessing.lesion_to_segmentation import mask_segmentation
   from neuro_lit.postprocessing.lesion_to_surface import project_mask_to_surface
   from pathlib import Path
   
   subjects = ['sub-01', 'sub-02', 'sub-03']
   
   for subject in subjects:
       # Mask segmentation
       mask_segmentation(
           input_seg=f'fastsurfer/{subject}/mri/aparc+aseg.mgz',
           lesion_mask=f'lit/{subject}/inpainting_mask.nii.gz',
           output_seg=f'fastsurfer/{subject}/mri/aparc+aseg+lesion.mgz'
       )
       
       # Mask surfaces
       for hemi in ['lh', 'rh']:
           project_mask_to_surface(
               lesion_mask=f'lit/{subject}/inpainting_mask.nii.gz',
               surface_file=f'fastsurfer/{subject}/surf/{hemi}.white',
               cortex_label=f'fastsurfer/{subject}/label/{hemi}.cortex.label',
               output_label=f'fastsurfer/{subject}/label/{hemi}.lesion.label'
           )

Integration with Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import subprocess
   
   def full_pipeline(subject_id, t1w_path, mask_path):
       """Run complete LIT + FastSurfer + masking pipeline"""
       
       # Step 1: LIT inpainting
       subprocess.run([
           'run-lit',
           '--input_image', t1w_path,
           '--mask_image', mask_path,
           '--output_directory', f'output/{subject_id}/lit',
           '--dilate', '2'
       ])
       
       # Step 2: FastSurfer (assumed to be run separately)
       # ...
       
       # Step 3: Mask segmentation
       subprocess.run([
           'python3', 'LIT/postprocessing/lesion_to_segmentation.py',
           '-i', f'output/{subject_id}/fastsurfer/mri/aparc+aseg.mgz',
           '-m', f'output/{subject_id}/lit/inpainting_volumes/inpainting_mask.nii.gz',
           '-o', f'output/{subject_id}/fastsurfer/mri/aparc+aseg+lesion.mgz'
       ])
       
       # Step 4: Mask surfaces
       for hemi in ['lh', 'rh']:
           subprocess.run([
               'python3', 'LIT/postprocessing/lesion_to_surface.py',
               '--inseg', f'output/{subject_id}/lit/inpainting_volumes/inpainting_mask.nii.gz',
               '--insurf', f'output/{subject_id}/fastsurfer/surf/{hemi}.white.preaparc',
               # ... other arguments
           ])

