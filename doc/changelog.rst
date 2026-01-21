Changelog
=========

All notable changes to LIT are documented here.

Version 0.5.1 (Current)
-----------------------

**Added:**

* PyPI package distribution (Test PyPI)
* Automatic model checkpoint downloading
* Platform-specific data directories using platformdirs
* CLI commands: ``run-lit``, ``inpaint-image``, ``lit-download-models``
* Progress bars for model downloads
* Containerized scripts for Docker and Singularity

**Fixed:**

* Model path resolution for pip installations
* Weight downloading from zenodo
* Compatibility with PyTorch 2.9+
* Memory efficiency in inference

**Changed:**

* Unified model storage location across installation methods
* Improved error messages and user feedback
* Updated documentation structure

Version 0.5.0
-------------

**Added:**

* Initial PyPI release preparation
* Postprocessing scripts for FastSurfer integration
* Lesion-to-segmentation masking
* Lesion-to-surface projection
* Lookup tables for lesion labeling

**Fixed:**

* Image orientation handling
* Intensity rescaling issues
* Mask dilation edge cases

**Changed:**

* Refactored project structure for pip packaging
* Improved modularity of inference pipeline
* Enhanced logging and error handling

Version 0.4.0
-------------

**Added:**

* Multi-view diffusion model (axial, coronal, sagittal)
* Automatic mask dilation
* Docker and Singularity support
* Training scripts for custom models

**Fixed:**

* GPU memory optimization
* Batch processing stability
* NIfTI header preservation

**Changed:**

* Improved inpainting quality
* Faster inference speed
* Better documentation

Version 0.3.0
-------------

**Added:**

* Initial public release
* Core inpainting functionality
* Pre-trained models
* Basic documentation
* Example scripts

**Features:**

* DDPM-based inpainting
* T1-weighted MRI support
* NIfTI format support
* GPU acceleration

Version 0.2.0 (Internal)
------------------------

**Added:**

* Proof-of-concept implementation
* Model architecture design
* Training pipeline
* Evaluation metrics

Version 0.1.0 (Internal)
------------------------

**Added:**

* Project initialization
* Literature review
* Initial experiments

Upcoming Features
-----------------

The following features are planned for future releases:

**Version 0.6.0 (Planned):**

* Direct FastSurfer integration
* Multi-modal support (T2, FLAIR)
* Improved surface masking
* Enhanced visualization tools
* Web-based demo

**Long-term Roadmap:**

* Real-time inpainting for surgical planning
* Multi-atlas segmentation support
* Automatic lesion detection
* Cloud deployment options
* GUI application

Migration Guides
----------------

From 0.4.x to 0.5.x
~~~~~~~~~~~~~~~~~~~

**Model Paths:**

Old location (git clone):

.. code-block:: text

   LIT/weights/model_*.pt

New location (unified):

.. code-block:: text

   ~/.local/share/LIT/weights/model_*.pt  # Linux
   ~/Library/Application Support/LIT/weights/model_*.pt  # macOS

**Running LIT:**

Old command:

.. code-block:: bash

   ./LIT/scripts/run_lit.sh --input T1w.nii.gz ...

New command (pip):

.. code-block:: bash

   run-lit --input_image T1w.nii.gz ...

**Python API:**

No breaking changes. All existing Python code should work unchanged.

From 0.3.x to 0.4.x
~~~~~~~~~~~~~~~~~~~

**Model Format:**

Models now require three separate checkpoints (axial, coronal, sagittal). Old single-model checkpoints are not compatible.

**Configuration:**

Configuration moved from YAML to command-line arguments for simplicity.

Deprecations
------------

**Version 0.5.x:**

* Direct model path specification may be deprecated in favor of automatic resolution
* Legacy configuration file support will be removed

**Version 0.6.x:**

* Old script locations (``./run_lit.sh``) may be deprecated in favor of CLI commands

Breaking Changes
----------------

**Version 0.5.0:**

* Model checkpoint format changed
* Configuration file structure changed
* Python package renamed from ``lit`` to ``LIT``

**Version 0.4.0:**

* Inference API signature changed
* Output directory structure modified

Bug Fixes
---------

**Version 0.5.1:**

* Fixed model download URLs
* Resolved path issues on Windows
* Fixed memory leak in batch processing
* Corrected affine matrix handling

**Version 0.5.0:**

* Fixed mask dilation boundary issues
* Resolved torch version compatibility
* Fixed NIfTI header corruption
* Corrected coordinate system transformations

**Version 0.4.0:**

* Fixed GPU out-of-memory errors
* Resolved image resampling artifacts
* Fixed multi-threading issues
* Corrected intensity normalization

Known Issues
------------

**Current:**

* Large images (>512^3) may cause memory issues
* Windows support is experimental
* Some edge cases in surface projection need improvement

**Workarounds:**

* For large images: Use CPU mode or reduce batch size
* For Windows: Use WSL2 or Docker
* For surface issues: Increase projection distance parameter

Contributing
------------

See the :doc:`contributing` guide for information on how to contribute to LIT.

Report bugs and request features on our `GitHub Issues page <https://github.com/Deep-MI/LIT/issues>`_.

