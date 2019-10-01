# mercs

Add a short description here!

## Description

A longer description of your project goes here...

## Installation

In order to set up the necessary environment:

1. create an environment `mercs` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate mercs
   ```
3. install `mercs` with:
   ```
   python setup.py install # or `develop`
   ```

Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
   
2. Create concrete dependencies as `environment.lock.yaml` for the exact reproduction of your
   environment with:
   ```
   conda env export -n mercs -f environment.lock.yaml
   ```
   
   _N.b.: For multi-OS development, consider using `--no-builds` during the export._
   
3. Update your current environment with respect to a new `environment.lock.yaml` using:
   ```
   conda env update -f environment.lock.yaml --prune
   ```
