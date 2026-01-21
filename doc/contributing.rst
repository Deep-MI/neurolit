Contributing
============

We welcome contributions to LIT! This guide will help you get started.

Getting Started
---------------

**Prerequisites:**

* Python >= 3.8
* Git
* Familiarity with PyTorch and brain MRI analysis

**Setup:**

.. code-block:: bash

   # Fork and clone the repository
   git clone https://github.com/YOUR_USERNAME/LIT.git
   cd LIT
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
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
       """
       Process a brain MRI image.
       
       Args:
           image_path: Path to input image
           output_dir: Directory for outputs
           device: Device to use ('cuda' or 'cpu')
           
       Returns:
           Dictionary containing results
           
       Raises:
           ValueError: If image_path doesn't exist
       """
       # Implementation...
       return results

Testing
~~~~~~~

Before submitting changes:

.. code-block:: bash

   # Run tests
   python3 -m pytest
   
   # Check code style
   black --check LIT/
   
   # Type checking
   mypy LIT/

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

Commit Messages
~~~~~~~~~~~~~~~

Write clear, descriptive commit messages:

.. code-block:: text

   Short summary (50 chars or less)
   
   More detailed explanation if needed. Wrap at 72 characters.
   
   - Bullet points are okay
   - Reference issues: Fixes #123

**Good examples:**

* ``Add support for multi-class lesion masks``
* ``Fix memory leak in inference pipeline``
* ``Update documentation for training script``

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

Testing Guidelines
------------------

Writing Tests
~~~~~~~~~~~~~

Add tests for new features:

.. code-block:: python

   # tests/test_inference.py
   import pytest
   from LIT.inference import run_inference
   
   def test_inference_basic():
       """Test basic inference functionality"""
       # Setup
       args = create_test_args()
       
       # Run
       result = run_inference(args)
       
       # Assert
       assert result is not None
       assert result['success'] == True

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/test_inference.py
   
   # Run with coverage
   pytest --cov=LIT tests/
   
   # Run verbose
   pytest -v

Documentation Guidelines
------------------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd doc
   make html  # Build HTML documentation
   make clean  # Clean build files

Documentation Style
~~~~~~~~~~~~~~~~~~~

* Use **reStructuredText** format
* Include **code examples** where helpful
* Add **cross-references** to related sections
* Keep **line length** reasonable (80-100 chars)
* Use **proper headings** hierarchy

**Example:**

.. code-block:: rst

   Section Title
   =============
   
   Subsection
   ----------
   
   Description with :class:`LIT.module.ClassName` reference.
   
   .. code-block:: python
   
      # Example code
      from LIT import something
      result = something()

Publishing to PyPI
------------------

See the main :doc:`contributing` page in the repository for detailed instructions on:

* Building packages
* Uploading to Test PyPI
* Publishing stable releases

Release Process
---------------

1. **Update version** in ``pyproject.toml``
2. **Update CHANGELOG.md** with changes
3. **Test thoroughly** on multiple systems
4. **Build package:** ``python3 -m build``
5. **Upload to Test PyPI** first
6. **Test installation** from Test PyPI
7. **Upload to PyPI** when ready
8. **Create git tag:** ``git tag -a v0.5.2 -m "Release 0.5.2"``
9. **Push tag:** ``git push origin v0.5.2``

Code Review
-----------

All contributions go through code review. Reviews focus on:

* **Correctness:** Does the code work as intended?
* **Style:** Does it follow project conventions?
* **Testing:** Are there adequate tests?
* **Documentation:** Is it well documented?
* **Performance:** Are there efficiency concerns?

Be responsive to feedback and willing to make changes!

Community
---------

Questions or Discussion
~~~~~~~~~~~~~~~~~~~~~~~

* **GitHub Issues:** For bugs and features
* **GitHub Discussions:** For general questions
* **Email:** clemens.pollak@dzne.de for sensitive matters

Code of Conduct
~~~~~~~~~~~~~~~

We expect all contributors to:

* Be respectful and professional
* Welcome newcomers
* Accept constructive criticism gracefully
* Focus on what's best for the project

Recognition
-----------

Contributors will be:

* Listed in the AUTHORS file
* Mentioned in release notes
* Acknowledged in papers if contributions are substantial

Thank You!
----------

Thank you for contributing to LIT! Every contribution, no matter how small, helps make the project better for everyone.

