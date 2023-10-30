# Documentation info

[Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html) 
is used to generate documentation.

Docstrings for this project follow a Google format. Click 
[here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for an example.

## How to generate the documentation

*Note: [`sphinx_rtd_theme`](https://github.com/readthedocs/sphinx_rtd_theme) 
needs to be installed for this to work.*

Run these commands from the project's root folder:

For Linux:
```bash
sphinx-apidoc -f -o docs/source src/
cd docs
make html
```

For Windows:
(Someone using Windows please put here a similar instruction for Windows)

## Guidelines

Base guidelines for docstrings in this project (this list might get updated 
in the future):

- Please put argument descriptions for any `__init__()` method in its class docstring 
(not in the method docstring).
