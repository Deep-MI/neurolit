Inference Module
================

.. automodule:: neurolit.inference
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The inference module exposes the inferer classes that implement neuroLIT's diffusion-based inpainting pipeline.

Key Concepts
------------

* :py:class:`neurolit.inference.InpaintingInferer`
* :py:class:`neurolit.inference.SliceWiseInpaintingInferer`
* :py:class:`neurolit.inference.TwoAndHalfDInpaintingInferer`

Use the command-line entry points defined in :py:mod:`neurolit.cli` and :py:mod:`neurolit.inpaint_image` when you prefer packaged invocations of these inferers.
