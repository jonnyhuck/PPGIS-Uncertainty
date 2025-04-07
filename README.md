# PPGIS Uncertainty

This repository contains the code  to support the Review of the paper _Embracing Uncertainty in Participatory GIS: Perceptions of tree planting in the English Lake District_, presented at GISRUK 2025.

- To reproduce the results, run `lr_combination.py`. This will output a raster containing six bands:
    1. Belief Trees
    1. Belief No Trees
    1. Plausibility Trees
    1. Plausibility No Trees
    1. Probability Trees
    1. Probability No Trees
- Generic methods for the functions are in `ling_rudd2.py`, if ou run this directly it will reproduce the example given in the method. 
- To reproduce the topic analysis, run `topics.py`

## Dependencies

- `lr_combination.py`: `geopandas` `rasterio`
- `ling_rudd2.py`: `none`
- `topics.py`: `pandas`, `nltk`<sup>*</sup>

<sup>*</sup> Note that you need to run some manual downloads once this is installed

To create an environment and install all of the dependencies, I recommend running the following commend using [anaconda](https://www.anaconda.com/):

```bash
conda create --name understandinggis --channel conda-forge --override-channels --yes python=3 geopandas rasterio nltk
```