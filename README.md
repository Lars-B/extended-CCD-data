# Data for extended CCD

This repository contains datasets that were used as examples in the extented CCD paper.

See the subfolders for specifics and references.

[![DOI](https://zenodo.org/badge/1146290961.svg)](https://doi.org/10.5281/zenodo.22663643)

## Running Python scripts

The Python scripts generate the MDS figures and other plots and data used in the paper.

To reproduce these results, create a virtual environment:

```bash
python -m venv recreate
```

Activate the environment:

```bash
source recreate/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

You can then run the two Python scripts to generate the MDS figures and the other plots and data used in the paper.
