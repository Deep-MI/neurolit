Postprocessing Module
=====================

The postprocessing module provides tools for integrating lesion masks into FastSurfer/FreeSurfer outputs.

Overview
--------

The recommended way to run postprocessing is via the ``lit-postprocessing`` command-line tool.

Command-Line Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   lit-postprocessing \
       --subject-id SUBJECT_ID \
       --subjects-dir /path/to/subjects_dir

This tool performs the following steps:
1. Maps the lesion mask into all relevant FastSurfer segmentation volumes (e.g., ``aparc+aseg.mgz``).
2. Generates anatomy reports (replaced, reduced, and adjacent labels).
3. Runs volumetric statistics (``segstats``) for the lesion-masked volumes.
4. Projects the lesion mask onto cortical surfaces and performs surface masking.

For more details on configuration, see ``segstats_config.json`` and ``surfstats_config.json`` in the ``neurolit/postprocessing`` directory.
