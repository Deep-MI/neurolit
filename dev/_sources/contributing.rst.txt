Contributing
============

We welcome contributions to neuroLIT! This guide will help you get started.

Getting Started
---------------

**Setup:**

.. code-block:: bash

   # Fork and clone the repository
   git clone https://github.com/YOUR_USERNAME/neurolit.git
   cd neurolit
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r pyproject.toml
   pip install -e .
   
   # Download models
   lit-download-models

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

If you find a bug, please open an issue with:

* Clear description of the problem
* Steps to reproduce
* Expected vs actual behavior
* Environment details (OS, Python version, GPU)
* Error messages and logs

Feature Requests
~~~~~~~~~~~~~~~~

We're open to new features! Please open an issue describing:

* The feature and its use case
* How it would work
* Why it would be valuable

Code Contributions
~~~~~~~~~~~~~~~~~~

1. **Fork the repository**
2. **Create a feature branch:** ``git checkout -b feature/my-feature``
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**
6. **Push and create a pull request**

Documentation
~~~~~~~~~~~~~

Documentation improvements are always welcome! This includes:

* Fixing typos or unclear sections
* Adding examples
* Improving API documentation
* Translating documentation

Development Guidelines
----------------------

Code Style
~~~~~~~~~~

* **Follow PEP 8** style guidelines
* **Use meaningful names** for variables and functions
* **Add docstrings** to all public functions and classes
* **Keep functions focused** and modular
* **Type hints** are encouraged

**Example:**

.. code-block:: python

   def process_image(
       image_path: str,
       output_dir: str,
       device: str = 'cuda'
   ) -> dict:
       """Process a brain MRI image.

       Parameters
       ----------
       image_path : str
           Path to input image
       output_dir : str
           Directory for outputs
       device : str, optional
           Device to use ('cuda' or 'cpu'), by default 'cuda'

       Returns
       -------
       dict
           Dictionary containing results

       Raises
       ------
       ValueError
           If image_path doesn't exist

       Examples
       --------
       >>> results = process_image(
       ...     image_path='brain.nii.gz',
       ...     output_dir='./outputs',
       ...     device='cuda'
       ... )
       >>> print(results.keys())
       """
       # Implementation...
       return results

Testing
~~~~~~~

Before submitting changes:

.. code-block:: bash
   
   # Check code style
   ruff check .

Documentation
~~~~~~~~~~~~~

When adding new features, update documentation:

.. code-block:: bash

   # Build documentation locally
   cd doc
   make html
   
   # View in browser
   firefox _build/html/index.html

Git Workflow
------------

Branch Naming
~~~~~~~~~~~~~

* ``feature/description`` - New features
* ``bugfix/description`` - Bug fixes
* ``doc/description`` - Documentation changes
* ``refactor/description`` - Code refactoring


Pull Requests
~~~~~~~~~~~~~

When creating a pull request:

1. **Clear title** describing the change
2. **Description** explaining what and why
3. **Reference issues** if applicable (Fixes #123)
4. **Screenshots** for UI changes
5. **Test results** if relevant

**Pull Request Template:**

.. code-block:: markdown

   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Added new tests for new features
   - [ ] Tested on real data
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No new warnings
   - [ ] Commit messages are clear


Thank You!
----------

Thank you for contributing to neuroLIT! Every contribution, no matter how small, helps make the project better for everyone.

