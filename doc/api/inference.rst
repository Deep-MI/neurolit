Inference Module
================

.. automodule:: lit.inference
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The inference module exposes the inferer classes that implement LIT's diffusion-based inpainting pipeline.

Key Concepts
------------

* :py:class:`lit.inference.InpaintingInferer`
* :py:class:`lit.inference.SliceWiseInpaintingInferer`
* :py:class:`lit.inference.TwoAndHalfDInpaintingInferer`

Use the command-line entry points defined in :py:mod:`lit.cli` and :py:mod:`lit.inpaint_image` when you prefer packaged invocations of these inferers.
