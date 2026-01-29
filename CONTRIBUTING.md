# Contributing to LIT

## Development Setup

### Prerequisites

- Python >= 3.8
- pip and setuptools
- build tools: `python3 -m pip install --upgrade build twine`

## Publishing to PyPI

### Publishing to Test PyPI (Beta Releases)

Test PyPI is useful for testing your package distribution before publishing to the main PyPI. This is where we host beta releases.

#### 1. Set up your Test PyPI account

First, create an account on [Test PyPI](https://test.pypi.org/account/register/) if you don't have one.

#### 2. Configure your Test PyPI credentials

Create or edit `~/.pypirc` with your Test PyPI credentials:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZw...  # Your Test PyPI API token
```

**Important:** Use API tokens instead of passwords for better security. Generate a token at [Test PyPI Account Settings](https://test.pypi.org/manage/account/#api-tokens).

#### 3. Update the version number

Edit `pyproject.toml` and increment the version number:

```toml
[project]
name = "neuro-lit"
version = "0.5.2"  # Increment this
```

#### 4. Clean previous builds

Remove any old build artifacts:

```bash
rm -rf dist/ build/ *.egg-info
```

#### 5. Build the package

Build the distribution files:

```bash
python3 -m build
```

This will create two files in the `dist/` directory:
- A source distribution (`.tar.gz`)
- A wheel distribution (`.whl`)

#### 6. Upload to Test PyPI

Upload the package to Test PyPI using twine:

```bash
python3 -m twine upload --repository testpypi dist/*
```

Or if you haven't configured `~/.pypirc`:

```bash
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --username __token__ --password YOUR_TEST_PYPI_TOKEN
```

#### 7. Test the installation

Wait a few minutes, then test installing from Test PyPI:

```bash
# Create a fresh virtual environment
python3 -m venv test_env
source test_env/bin/activate

# Install from Test PyPI
pip install -i https://test.pypi.org/simple/ neuro-lit

# Download model checkpoints
lit-download-models

# Test the installation
run-lit --help

# Test with actual data (if available)
# run-lit --input_image test.nii.gz --mask_image mask.nii.gz --output_directory test_output

# Deactivate and clean up
deactivate
rm -rf test_env
```

### Publishing to PyPI (Stable Releases)

Once you've tested the package on Test PyPI and are ready for a stable release:

#### 1. Ensure version is updated

Make sure `pyproject.toml` has the correct version number for the stable release.

#### 2. Build the package

```bash
rm -rf dist/ build/ *.egg-info
python3 -m build
```

#### 3. Upload to PyPI

```bash
python3 -m twine upload dist/*
```

Or with explicit credentials:

```bash
python3 -m twine upload dist/* --username __token__ --password YOUR_PYPI_TOKEN
```

#### 4. Create a Git tag

Tag the release in Git:

```bash
git tag -a v0.5.2 -m "Release version 0.5.2"
git push origin v0.5.2
```

### Troubleshooting

#### Issue: "File already exists"

If you get an error about a file already existing, you need to increment the version number in `pyproject.toml`. PyPI doesn't allow re-uploading the same version.

#### Issue: Dependencies not installing

Test PyPI doesn't have all the packages that main PyPI has. When installing from Test PyPI, you may need to allow pip to fall back to main PyPI for dependencies:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ neuro-lit
```

#### Issue: Package not found immediately after upload

It can take a few minutes for the package to be indexed and available on Test PyPI or PyPI.

#### Issue: Script errors or file not found

If you encounter errors, ensure that:

1. **`platformdirs` is installed**: Required for determining the correct data directory
2. **Models are downloaded**: Run `lit-download-models` or let them download automatically on first use
3. **All dependencies are installed**: Check `requirements.txt` and `pyproject.toml`

The codebase now uses consistent paths for both pip and git installations:
- Model checkpoints are stored in platform-specific locations via `platformdirs`
  - Linux: `~/.local/share/neuro-lit/weights`
  - macOS: `~/Library/Application Support/neuro-lit/weights`
  - Windows: `C:\Users\<user>\AppData\Local\Deep-MI\neuro-lit\weights`

Make sure to test both installation methods before publishing:

```bash
# Test pip installation
pip install -i https://test.pypi.org/simple/ neuro-lit
lit-download-models  # Optional: pre-download models
run-lit --input_image test.nii.gz --mask_image mask.nii.gz --output_directory test_output

# Test git clone (uses same model location)
git clone https://github.com/Deep-MI/neuro-lit.git
cd neuro-lit
lit-download-models  # Optional: pre-download models
./LIT/scripts/run_lit.sh --input_image test.nii.gz --mask_image mask.nii.gz --output_directory test_output
```

### Quick Reference Commands

```bash
# Clean, build, and upload to Test PyPI in one go
rm -rf dist/ build/ *.egg-info && \
python3 -m build && \
python3 -m twine upload --repository testpypi dist/*

# Clean, build, and upload to PyPI in one go
rm -rf dist/ build/ *.egg-info && \
python3 -m build && \
python3 -m twine upload dist/*
```

## Available CLI Commands

After installing via pip, the following command-line tools are available:

### `run-lit`
Main command to run LIT with the full pipeline (wrapper around `run_lit.sh`).

```bash
run-lit --input_image T1w.nii.gz --mask_image lesion_mask.nii.gz --output_directory output_dir
```

### `inpaint-image`
Direct access to the core inpainting functionality.

```bash
inpaint-image --input_image T1w.nii.gz --mask_image mask.nii.gz --out_dir output_dir
```

### `lit-download-models`
Download the required model checkpoints. This is recommended to run once after installation to avoid delays on first use.

```bash
lit-download-models
```

This command will:
- Download models to a consistent platform-specific location using `platformdirs`
  - Linux: `~/.local/share/LIT/weights`
  - macOS: `~/Library/Application Support/LIT/weights`
  - Windows: `C:\Users\<user>\AppData\Local\Deep-MI\LIT\weights`
- Show real-time progress bars with download speed and ETA
- Verify existing models and skip already downloaded ones
- Provide clear success/failure messages

## Code Style and Standards

### Python Code

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular

### Testing

Before submitting changes, ensure all tests pass:

```bash
python3 -m pytest
```

## Making Changes

1. Create a fork for your feature or bugfix
2. Make your changes
3. Test your changes thoroughly
4. Update documentation (README.md, etc.) as needed
5. Commit with clear, descriptive messages
6. Create a pull request to the dev branch

## Project Structure

```
neuro-lit/
├── neuro_lit/             # Main package directory
│   ├── scripts/           # Shell scripts including run_lit.sh
│   ├── utils/             # Utility modules
│   ├── inpaint_image.py   # Core inpainting functionality
│   └── cli.py             # Command-line interface
├── tests/                 # Test files
├── pyproject.toml         # Package configuration
├── requirements.txt       # Development dependencies
└── README.md              # User documentation
```

## Documentation

### Building Documentation

LIT uses Sphinx for documentation. To build the documentation locally:

```bash
# Install documentation dependencies
pip install -r doc/requirements.txt

# Build HTML documentation
cd doc
make html

# View in browser
firefox _build/html/index.html  # Or your preferred browser
```

### Documentation Guidelines

When contributing to documentation:

1. **Use reStructuredText (.rst) format** for documentation files
2. **Include code examples** where helpful
3. **Add docstrings** to all public functions and classes
4. **Update API docs** when changing function signatures
5. **Build and preview** before submitting

**Docstring Format:**

Use NumPy-style docstrings (example from: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html):

```python
def module_level_function(param1, param2=None, *args, **kwargs):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Parameters`` section.
    The name of each parameter is required. The type and description of each
    parameter is optional, but should be included if not obvious.

    If \*args or \*\*kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name : type
            description

            The description may span multiple lines. Following lines
            should be indented to match the first line of the description.
            The ": type" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : :obj:`str`, optional
        The second parameter.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    bool
        True if successful, False otherwise.

    The return type is not optional. The ``Returns`` section may span
    multiple lines and paragraphs. Following lines should be indented to
    match the first line of the description.

    The ``Returns`` section supports any reStructuredText formatting,
    including literal blocks::

        {
            'param1': param1,
            'param2': param2
        }

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
    ValueError
        If `param2` is equal to `param1`.

    """
    pass
```

**Adding New Documentation Pages:**

1. Create a new `.rst` file in the `doc/` directory
2. Add content using reStructuredText syntax
3. Add the page to a `toctree` directive in `index.rst` or parent page
4. Build and preview the documentation

**Documentation Structure:**

```
doc/
├── index.rst            # Main page
├── installation.rst     # Installation guide
├── usage.rst           # Usage guide
├── integration.rst     # FastSurfer integration
├── training.rst        # Training guide
├── contributing.rst    # Contributing guide
├── api/                # API reference
│   └── ...
└── ...
```

For more details, see `doc/README.md`.

## Questions or Issues?

If you have questions or run into issues, please open an issue on the [GitHub repository](https://github.com/Deep-MI/LIT).

