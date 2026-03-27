Downstream Analysis with Lesion Exclusion
==========================================

The lesion mask and the anatomy reports produced by ``lit-postprocessing`` (see :doc:`usage`) can be used to **exclude lesion-affected regions from downstream statistical analyses**. This applies both to surface-based analyses (vertex-wise metrics such as cortical thickness) and to volumetric analyses (region-of-interest volumes from parcellation or segmentation).

The same principle underlies both approaches: identify which regions are affected by the lesion, set their values to ``NaN`` or drop them from the table, and let your statistical test handle the missing data. This prevents lesion-driven measurement artefacts from inflating or deflating group statistics, reproducibility metrics, or any other comparison.

---

Surface-Based Analysis
-----------------------

The workflow below uses cortical thickness as the example measure, but the same approach applies to any per-vertex surface metric (curvature, sulcal depth, myelination, etc.).

**Overview**

1. Run neuroLIT inpainting and postprocessing to obtain per-subject lesion surface labels
2. Project the cortical measure onto a common surface template (``fsaverage``) with ``mris_preproc``
3. Set lesion vertices to ``NaN`` in the stacked surface data
4. Run your statistical analysis; NaN vertices are excluded vertex-wise

---

Step 1 — Obtain Lesion Surface Labels
--------------------------------------

Run the neuroLIT postprocessing pipeline to generate FreeSurfer ``.label`` files marking lesion vertices on the cortical surface for each subject:

.. code-block:: bash

   lit-postprocessing \
       --subject-id SUBJECT_ID \
       --subjects-dir /path/to/subjects_dir

This produces ``label/{lh,rh}.lesion.label`` inside each subject directory. These label files are in the subject's native surface space and can be projected to fsaverage together with the measure of interest.

---

Step 2 — Extract Regional Statistics (Optional)
-------------------------------------------------

Standard FreeSurfer tools can extract parcellation-level statistics before or after lesion exclusion, depending on whether the postprocessing step has already updated the annotation files:

.. code-block:: bash

   # Cortical thickness per parcellation (DKT atlas), per hemisphere
   aparcstats2table --subjects $SUBJECTS \
       --hemi lh --measure thickness \
       --parc aparc.DKTatlas.mapped \
       --tablefile lh.thickness.txt

   aparcstats2table --subjects $SUBJECTS \
       --hemi rh --measure thickness \
       --parc aparc.DKTatlas.mapped \
       --tablefile rh.thickness.txt

   # Subcortical volumes
   asegstats2table --subjects $SUBJECTS \
       --meas volume \
       --tablefile aseg.volume.txt \
       --delimiter space \
       --stats aseg.stats

---

Step 3 — Project to Common Surface Template
--------------------------------------------

Project individual subject surface maps onto ``fsaverage`` using ``mris_preproc``. A spatial smoothing kernel (e.g. 15 mm FWHM) is typically applied:

.. code-block:: bash

   mris_preproc \
       --s SUBJECT1 --s SUBJECT2 ... \
       --meas thickness \
       --fwhm 15 \
       --target fsaverage \
       --hemi lh \
       --out lh.thickness.fwhm15.stack.mgh

   mris_preproc ... --hemi rh --out rh.thickness.fwhm15.stack.mgh

The output ``.mgh`` files are vertex-wise stacks with shape ``(n_subjects, n_vertices)``.

---

Step 4 — Mask Lesion Vertices
-------------------------------

Before running any statistical analysis, set lesion-affected vertices to ``NaN`` so they are excluded vertex-wise:

.. code-block:: python

   import nibabel as nib
   import numpy as np

   # Load stacked surface data
   thickness_stack = nib.load("lh.thickness.fwhm15.stack.mgh").get_fdata().squeeze()
   # shape: (n_subjects, n_vertices)

   # Load per-subject lesion label files and mask affected vertices
   for i, lesion_label_path in enumerate(lesion_label_paths):
       lesion_vertices = nib.freesurfer.io.read_label(lesion_label_path)
       thickness_stack[i, lesion_vertices] = np.nan

   # Restrict to cortex vertices only
   cortex_vertices = nib.freesurfer.io.read_label("fsaverage/label/lh.cortex.label")
   thickness_cortex = thickness_stack[:, cortex_vertices]
   # NaN vertices are excluded automatically by numpy/scipy functions

The ``lh.cortex.label`` file is part of the standard fsaverage distribution and defines the non-medial-wall region.

---

Step 5 — Run Your Statistical Analysis
----------------------------------------

With lesion vertices masked to ``NaN``, any vertex-wise analysis that handles missing data will automatically exclude them. Examples:

**Test-retest reproducibility (ICC)**

Intraclass Correlation Coefficient (ICC type A-1, two-way mixed absolute agreement) at each vertex:

.. code-block:: python

   n_subjects = thickness_stack1.shape[0]
   k = 2  # number of sessions

   icc_map = np.full(n_cortex_vertices, np.nan)
   for v in range(n_cortex_vertices):
       col1 = thickness_stack1[:, v]
       col2 = thickness_stack2[:, v]
       valid = ~np.isnan(col1) & ~np.isnan(col2)
       if valid.sum() < 3:
           continue
       data = np.stack([col1[valid], col2[valid]], axis=1)
       n = data.shape[0]
       grand_mean = data.mean()
       ss_total = ((data - grand_mean) ** 2).sum()
       ss_rows = k * ((data.mean(axis=1) - grand_mean) ** 2).sum()
       ss_cols = n * ((data.mean(axis=0) - grand_mean) ** 2).sum()
       ss_error = ss_total - ss_rows - ss_cols
       msr = ss_rows / (n - 1)
       msc = ss_cols / (k - 1)
       mse = ss_error / ((n - 1) * (k - 1))
       icc_map[v] = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

**Group differences (t-test)**

.. code-block:: python

   from scipy.stats import ttest_ind

   pval_map = np.full(n_cortex_vertices, np.nan)
   for v in range(n_cortex_vertices):
       g1 = group1_stack[:, v]
       g2 = group2_stack[:, v]
       valid1, valid2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]
       if len(valid1) < 3 or len(valid2) < 3:
           continue
       _, pval_map[v] = ttest_ind(valid1, valid2)

---

Step 6 — Visualize on the Cortical Surface (Optional)
-------------------------------------------------------

Surface maps can be rendered using `WhipperSnappy <https://github.com/deep-mi/whippersnappy>`_:

.. code-block:: python

   from whippersnappy.core import snap4

   snap4(
       lhoverlaypath="lh.result.fwhm15.mgh",
       rhoverlaypath="rh.result.fwhm15.mgh",
       fthresh=0.8,
       fmax=1.0,
       sdir="/path/to/fsaverage",
       outpath="result_surface.png"
   )

Alternatively, the maps can be loaded into standard neuroimaging viewers (FreeView, FSLeyes) for interactive inspection.

---

Volumetric Analysis
--------------------

For region-of-interest (ROI) analyses based on segmentation volumes, the **lesion anatomy reports** generated by ``lit-postprocessing`` identify which parcellation regions are affected by the lesion. This makes it straightforward to exclude unreliable measurements before running group comparisons, correlations, or other statistics on volumetric data.

Anatomy Reports
~~~~~~~~~~~~~~~

For each subject, ``lit-postprocessing`` writes plain-text anatomy reports alongside the modified segmentation files:

* ``stats/aparc.DKTatlas+aseg.lesion_report.txt`` — cortical parcellation and subcortical structures (DKT atlas)
* ``stats/aseg.lesion_report.txt`` — FreeSurfer aseg segmentation

Each report classifies every brain region into one of three categories:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Category
     - Meaning
   * - **Replaced**
     - Region is fully covered by the lesion. The entire structure has been relabelled; its volume is no longer meaningful.
   * - **Reduced**
     - Region is partially covered. The measured volume is smaller than the true anatomy; the degree of reduction depends on lesion overlap.
   * - **Adjacent**
     - Region touches the lesion boundary but has no voxel overlap. Its volume is intact, but proximity may introduce inpainting artefacts at the boundary.

The appropriate exclusion threshold depends on your analysis goals. A conservative approach excludes all three categories; a more permissive one retains adjacent regions and only drops replaced or reduced ones.

Excluding Affected Regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general workflow is:

1. Parse the per-subject anatomy report to obtain the set of affected region names
2. Build a subject × region table of volumes (e.g. from ``asegstats2table`` / ``aparcstats2table`` output)
3. For each subject, set the volumes of affected regions to ``NaN``
4. Run your analysis column-wise, dropping or ignoring NaN entries

A minimal Python example:

.. code-block:: python

   import pandas as pd
   import numpy as np

   def parse_lesion_report(report_path, categories=("Replaced", "Reduced")):
       """Return region names listed under the given categories in a lesion report."""
       affected = set()
       current_category = None
       with open(report_path) as f:
           for line in f:
               line = line.strip()
               if not line or line.startswith("#"):
                   continue
               # Section headers contain the category name
               for cat in categories:
                   if cat in line:
                       current_category = cat
                       break
               else:
                   if current_category is not None:
                       # Lines under a section are "label_id  region_name"
                       parts = line.split(None, 1)
                       if len(parts) == 2 and parts[0].isdigit():
                           affected.add(parts[1])
       return affected

   # Load volumetric stats table (rows = subjects, columns = regions)
   volumes = pd.read_csv("aseg_volumes.csv", index_col="subject")

   # Mask affected regions per subject
   for subject in volumes.index:
       report = f"subjects/{subject}/stats/aseg.lesion_report.txt"
       affected_regions = parse_lesion_report(report, categories=("Replaced", "Reduced"))
       cols_to_mask = [c for c in volumes.columns if c in affected_regions]
       volumes.loc[subject, cols_to_mask] = np.nan

   # Each column now contains NaN for subjects where that region was lesion-affected.
   # Standard pandas/scipy functions skip NaN by default.
   group_means = volumes.groupby("group").mean()

Choosing Which Categories to Exclude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Replaced regions** should almost always be excluded — the volume reflects the lesion label, not the anatomy.
* **Reduced regions** are typically excluded as well, since the measured volume is artificially smaller than the true value.
* **Adjacent regions** can often be retained for volume analyses (the voxel count is unaffected), but may warrant exclusion in studies where boundary effects are a concern (e.g. intensity-based measures, or when the inpainting boundary falls within a thin structure).

For reproducibility studies, a reasonable default is to exclude replaced and reduced regions and flag adjacent ones.

---

Dependencies
------------

* **FreeSurfer**: ``mris_preproc``, ``asegstats2table``, ``aparcstats2table``, fsaverage template
* **Python**: ``nibabel``, ``numpy``, ``scipy``, ``pandas``
* **WhipperSnappy** (optional, for surface visualization): ``pip install whippersnappy``
