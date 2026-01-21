# Documentation Quick Start

This guide will help you quickly build and view the LIT documentation.

## Quick Build

```bash
# Install dependencies
pip install -r doc/requirements.txt

# Build HTML documentation
cd doc && make html

# View in browser (Linux)
firefox _build/html/index.html

# View in browser (macOS)
open _build/html/index.html

# View in browser (Windows)
start _build/html/index.html
```

## Common Tasks

### Clean and Rebuild

```bash
cd doc
make clean
make html
```

### Check for Broken Links

```bash
cd doc
make linkcheck
```

### Build PDF (requires LaTeX)

```bash
cd doc
make latexpdf
```

### Build Other Formats

```bash
cd doc
make epub        # EPUB format
make singlehtml  # Single HTML page
make man         # Man pages
```

## Troubleshooting

### "sphinx-build not found"

Install Sphinx:
```bash
pip install sphinx
```

### Import Errors

Make sure LIT is installed:
```bash
pip install -e .
```

### Missing Extensions

Install all documentation dependencies:
```bash
pip install -r doc/requirements.txt
```

## Directory Structure

```
doc/
├── _build/           # Build output (not in git)
├── _static/          # Static files (CSS, images)
├── _templates/       # Custom templates
├── api/              # API documentation
├── conf.py           # Sphinx configuration
├── index.rst         # Main page
├── *.rst             # Other pages
├── Makefile          # Unix build script
├── make.bat          # Windows build script
└── requirements.txt  # Documentation dependencies
```

## Continuous Integration

The documentation is automatically built by:

- **GitHub Actions**: On every push/PR (see `.github/workflows/doc.yml`)
- **Read the Docs**: Automatically from the repository (see `.readthedocs.yml`)

## Read the Docs

To enable Read the Docs:

1. Go to https://readthedocs.org/
2. Import the LIT repository
3. The configuration is already in `.readthedocs.yml`
4. Docs will auto-build on every commit

## GitHub Pages

To deploy to GitHub Pages:

1. Uncomment the deploy step in `.github/workflows/doc.yml`
2. Enable GitHub Pages in repository settings
3. Select "gh-pages" branch as source
4. Docs will auto-deploy on main branch commits

## Writing Documentation

### Add a New Page

1. Create `doc/my_page.rst`:
   ```rst
   My Page Title
   =============
   
   Content goes here.
   ```

2. Add to `doc/index.rst`:
   ```rst
   .. toctree::
      :maxdepth: 2
      
      my_page
   ```

3. Build and view:
   ```bash
   cd doc && make html
   ```

### Code Examples

```rst
.. code-block:: python

   from LIT import something
   result = something()
```

### Cross-References

```rst
See :doc:`installation` for details.
See :class:`LIT.module.ClassName` for API.
See :func:`LIT.module.function` for usage.
```

## Questions?

- See `doc/README.md` for detailed information
- See `CONTRIBUTING.md` for contribution guidelines
- Open an issue on GitHub for help

