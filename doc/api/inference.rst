Inference Module
================

.. automodule:: neuro_lit.inference
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The inference module exposes the inferer classes that implement LIT's diffusion-based inpainting pipeline.

Key Concepts
------------

* :py:class:`neuro_lit.inference.InpaintingInferer`
* :py:class:`neuro_lit.inference.SliceWiseInpaintingInferer`
* :py:class:`neuro_lit.inference.TwoAndHalfDInpaintingInferer`

Use the command-line entry points defined in :py:mod:`neuro_lit.cli` and :py:mod:`neuro_lit.inpaint_image` when you prefer packaged invocations of these inferers.
