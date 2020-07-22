#!/bin/bash

set -e

# Deactivate the travis-provided venv and setup conda
deactivate

if [[ -f "$HOME/miniconda/bin/conda" ]]; then
        echo "Skip install conda [cached]"
else
    # By default, travis caching mechanism creates an empty dir in the
    # beginning of the build, but conda installer aborts if it finds an
    # existing folder, so let's just remove it:
    rm -rf "$HOME/miniconda"
    
    # Use miniconda
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
fi

# Update path
export PATH=$HOME/miniconda/bin:$PATH

# Update conda
conda update --yes conda

# Create a local environment from the environment.yml file.

echo "Recreate conda environment in Travis servers"
conda env create -f dependencies-deploy.yaml -n mercs
echo "Recreation done"

# Conda activate does not work, use source instead.
echo "Add development packages"
source activate mercs
conda env update -n mercs -f dependencies-develop.yaml
echo "Development packages OK. Environment active."

travis-cleanup() {
    printf "Cleaning up environments ... "  # printf avoids new lines
    if [[ "$DISTRIB" == "conda" ]]; then
    # Force the env to be recreated next time, for build consistency
        source deactivate
        conda remove -n mercs --all --yes
        rm -rf ./.conda
    fi
    echo "DONE"
}

