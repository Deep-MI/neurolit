Inference Module
================

.. automodule:: LIT.inference
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The inference module exposes the inferer classes that implement LIT's diffusion-based inpainting pipeline.

Key Concepts
------------

* :py:class:`LIT.inference.InpaintingInferer`
* :py:class:`LIT.inference.SliceWiseInpaintingInferer`
* :py:class:`LIT.inference.TwoAndHalfDInpaintingInferer`

Use the command-line entry points defined in :py:mod:`LIT.cli` and :py:mod:`LIT.inpaint_image` when you prefer packaged invocations of these inferers.
