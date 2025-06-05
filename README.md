# PPGIS Uncertainty

This repository contains the code to adopt the method and reproduce the examples given in the paper support the Review of the papers _Decision Making under Uncertainty: Increasing the Impact of Public Participatory GIS_ (accepted for publication in the International Journal of Geographical Information Science) and [_Embracing Uncertainty in Participatory GIS: Perceptions of tree planting in the English Lake District_](https://zenodo.org/records/15309828), published in the proceedings of GISRUK 2025.

- To reproduce the results, run `lr_combination.py`. This will output a raster containing six bands:
    1. Belief Trees
    1. Belief No Trees
    1. Plausibility Trees
    1. Plausibility No Trees
    1. Probability Trees
    1. Probability No Trees
- Generic methods for the functions are in `ling_rudd2.py`, if you run this directly it will reproduce the example given in the method. 
- To reproduce the topic analysis, run `topics.py`

The PPGIS dataset collected using [map-me](https://map-me.org/) is located in `data/blobs.csv`. The associated text is in `data/map-me_answers_23-10-2023_12-39.csv`, and the processed output from this text is in `./data/answers_with_terms.csv`.

Also contained in the `data/` directory is a Shapefile containing geometries for the lakes, which is &copy; 2024 OpenStreetMap Contributors.

The code used for the version using [Focal Area Bias](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2440048) and the sensitivity analyses are located in the `supplementaries` directory.

## Dependencies

- `lr_combination.py`: `geopandas` `rasterio`
- `ling_rudd2.py`: `none`
- `topics.py`: `pandas`, `nltk`<sup>*</sup>

<sup>*</sup> Note that you need to run some manual downloads once this is installed, see [here](https://www.nltk.org/install.html).

To create an environment and install all of the dependencies, I recommend running the following commend using [anaconda](https://www.anaconda.com/):

```bash
conda create --name ppgis --channel conda-forge --override-channels --yes python=3 geopandas rasterio nltk
```