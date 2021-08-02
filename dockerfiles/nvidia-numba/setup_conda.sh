#! /usr/bin/env sh

conda create --name bandits-numba jupyter
conda config --add channels conda-forge
conda config --set channel_priority strict
mamba install numba numpy pandas -c conda-forge holoviews hvplot
