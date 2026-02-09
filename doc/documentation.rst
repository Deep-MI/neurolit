Documentation Guide
===================

This guide explains how to build and contribute to the LIT documentation.

Building the Documentation
--------------------------

Prerequisites
~~~~~~~~~~~~~

Install the documentation dependencies from the package:

.. code-block:: bash

   # Using uv (recommended)
   uv pip install -e ".[doc]"

   # Or using pip
   pip install -e ".[doc]"

Build HTML
~~~~~~~~~~

.. code-block:: bash

   cd doc
   make html

The built documentation will be in ``doc/_build/html/``. Open ``doc/_build/html/index.html`` in your browser.

Clean Build Files
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make clean

Writing Documentation
---------------------

LIT uses Sphinx with reStructuredText (reST) and MyST-Parser for Markdown support.

API Documentation
~~~~~~~~~~~~~~~~~

API documentation is auto-generated from Python docstrings using ``sphinx.ext.autodoc``. We follow the NumPy docstring style.

Adding New Pages
~~~~~~~~~~~~~~~~

1. Create a new ``.rst`` (or ``.md``) file in the ``doc/`` directory.
2. Add the page to the ``toctree`` directive in ``index.rst``.
3. Rebuild the documentation to preview changes.
