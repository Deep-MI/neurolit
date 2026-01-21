Citation
========

If you use LIT in your research, please cite our paper.

BibTeX
------

.. code-block:: bibtex

   @article{pollak2024fastsurfer,
     title={FastSurfer-LIT: Lesion Inpainting Tool for Whole Brain MRI Segmentation with Tumors, Cavities and Abnormalities},
     author={Pollak, Clemens and Kuegler, David and Bauer, Tobias and Rueber, Theodor and Reuter, Martin},
     journal={Imaging Neuroscience},
     year={2024},
     doi={10.1162/imag_a_00446},
     url={https://doi.org/10.1162/imag_a_00446}
   }

APA Format
----------

Pollak, C., Kuegler, D., Bauer, T., Rueber, T., & Reuter, M. (2024). FastSurfer-LIT: Lesion Inpainting Tool for Whole Brain MRI Segmentation with Tumors, Cavities and Abnormalities. *Imaging Neuroscience*. https://doi.org/10.1162/imag_a_00446

Plain Text
----------

Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M. FastSurfer-LIT: Lesion Inpainting Tool for Whole Brain MRI Segmentation with Tumors, Cavities and Abnormalities. Imaging Neuroscience. 2024. doi:10.1162/imag_a_00446

Related Publications
--------------------

If you also use FastSurfer, please cite:

.. code-block:: bibtex

   @article{henschel2020fastsurfer,
     title={FastSurfer - A fast and accurate deep learning based neuroimaging pipeline},
     author={Henschel, Leonie and Conjeti, Sailesh and Estrada, Santiago and Diers, Kersten and Fischl, Bruce and Reuter, Martin},
     journal={NeuroImage},
     volume={219},
     pages={117012},
     year={2020},
     doi={10.1016/j.neuroimage.2020.117012}
   }

Acknowledgments
---------------

Development and Testing
~~~~~~~~~~~~~~~~~~~~~~~

LIT was developed by:

* **Clemens Pollak** - Lead Developer
* **David Kuegler** - Core Development
* **Tobias Bauer** - Testing and Validation
* **Theodor Rueber** - Clinical Input
* **Martin Reuter** - Principal Investigator

The development team would like to thank all contributors and users who have provided feedback and bug reports.

Funding
~~~~~~~

This work was supported by:

* German Research Foundation (DFG)
* Helmholtz Association
* European Research Council (ERC)

Institutions
~~~~~~~~~~~~

* **German Center for Neurodegenerative Diseases (DZNE), Bonn, Germany**
* **University of Bonn, Germany**
* **A.A. Martinos Center for Biomedical Imaging, MGH, Harvard Medical School, USA**

Data and Resources
~~~~~~~~~~~~~~~~~~

We acknowledge the use of:

* Public brain MRI datasets for training and validation
* Open-source software libraries (PyTorch, nibabel, MONAI)
* High-performance computing resources

Using LIT in Publications
--------------------------

When describing LIT in your methods section, we suggest:

**Example Methods Description:**

    *"Lesion inpainting was performed using the Lesion Inpainting Tool (LIT) [Pollak et al., 2024], a deep learning-based method utilizing a denoising diffusion probabilistic model. The T1-weighted MRI and corresponding lesion mask were processed using LIT version 0.5.1 with default parameters, including 2-voxel mask dilation. The inpainted images were then used for [downstream analysis]."*

Please include:

* The LIT version used
* Key parameters (especially if non-default)
* How the inpainted images were used in your analysis

Share Your Work
---------------

We'd love to hear about your research using LIT! If you publish a paper:

* Let us know by emailing clemens.pollak@dzne.de
* We'll feature it on our website and documentation
* Your work helps demonstrate LIT's impact

Community Contributions
-----------------------

If you've contributed code or documentation to LIT that is used in published research, you'll be acknowledged in:

* The AUTHORS file
* Release notes
* This documentation

Substantial contributions may warrant co-authorship on method papers.

License
-------

LIT is released under the MIT License. See :doc:`license` for details.

You are free to:

* Use LIT in commercial and academic research
* Modify the source code
* Distribute the software

Just remember to:

* Cite the paper in publications
* Include the license and copyright notice
* Acknowledge the developers

Questions About Citation
------------------------

If you have questions about how to cite LIT or acknowledge contributions:

* Open an issue on `GitHub <https://github.com/Deep-MI/LIT/issues>`_
* Email the developers: clemens.pollak@dzne.de

We're happy to provide guidance on proper attribution!

