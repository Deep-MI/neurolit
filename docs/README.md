# LIT Documentation

This directory contains the Sphinx documentation source for LIT.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser.

### Clean Build Files

```bash
cd docs
make clean
```

### Other Output Formats

```bash
make latexpdf  # Build PDF (requires LaTeX)
make epub      # Build EPUB
make help      # See all available formats
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── installation.rst     # Installation guide
├── usage.rst           # Usage guide
├── integration.rst     # FastSurfer integration
├── training.rst        # Training guide
├── contributing.rst    # Contributing guide
├── changelog.rst       # Changelog
├── citation.rst        # Citation information
├── license.rst         # License information
├── api/                # API reference
│   ├── modules.rst
│   ├── cli.rst
│   ├── inference.rst
│   ├── inpainting.rst
│   ├── data.rst
│   ├── networks.rst
│   ├── postprocessing.rst
│   └── utils.rst
├── _static/            # Static files (CSS, images)
├── _templates/         # Custom templates
└── requirements.txt    # Documentation dependencies
```

## Writing Documentation

### reStructuredText Basics

Headings:
```rst
Main Title
==========

Section
-------

Subsection
~~~~~~~~~~
```

Links:
```rst
`Link text <URL>`_
:doc:`other_page`
:class:`LIT.module.ClassName`
```

Code blocks:
```rst
.. code-block:: python

   import LIT
   result = LIT.function()
```

### Adding New Pages

1. Create a new `.rst` file
2. Add content using reStructuredText
3. Add the page to a `toctree` directive in `index.rst` or another parent page

### API Documentation

API documentation is auto-generated from Python docstrings. Use Google or NumPy style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

## Previewing Changes

After making changes, rebuild the docs:

```bash
cd docs
make clean
make html
firefox _build/html/index.html  # Or your preferred browser
```

## Publishing Documentation

Documentation can be published to:

- **Read the Docs**: Automatically builds from GitHub
- **GitHub Pages**: Use `make html` and deploy `_build/html/`
- **Self-hosted**: Copy `_build/html/` to web server

## Troubleshooting

### Import Errors

If you get import errors when building:

```bash
# Make sure LIT is importable
pip install -e ..
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### Build Warnings

Sphinx is strict about formatting. Fix warnings to ensure quality:

- Check cross-reference syntax
- Verify indentation in code blocks
- Ensure all referenced files exist

## Contributing

See the main `CONTRIBUTING.md` in the repository root for general contribution guidelines.

For documentation-specific contributions:

1. Follow reStructuredText best practices
2. Include code examples where helpful
3. Keep line length reasonable (80-100 characters)
4. Build and preview before submitting
5. Fix any Sphinx warnings

## Questions?

- Open an issue on GitHub
- Contact the development team
- See [Sphinx documentation](https://www.sphinx-doc.org/)

