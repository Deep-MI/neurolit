# Contributing to neuroLIT


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

## Documentation

### Building Documentation

neuroLIT uses Sphinx for documentation. To build the documentation locally:

```bash
# Install documentation dependencies
pip install -r doc/requirements.txt

# Build HTML documentation
sphinx-build doc doc-build

# View in browser
firefox doc-build/index.html  # Or your preferred browser
```

### Documentation Guidelines

When contributing to documentation:

1. **Use reStructuredText (.rst) format** for documentation files
2. **Include code examples** where helpful
3. **Add docstrings** to all public functions and classes (NumpyStyle)
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

If you have questions or run into issues, please open an issue on the [GitHub repository](https://github.com/Deep-MI/neurolit).

