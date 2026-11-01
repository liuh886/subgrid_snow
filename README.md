# subgrid_snow

[![Paper: 10.1016/j.coldregions.2025.104580](https://img.shields.io/badge/DOI-10.1016%2Fj.coldregions.2025.104580-blue.svg)](https://doi.org/10.1016/j.coldregions.2025.104580)
[![Dataset: Zenodo](https://img.shields.io/badge/Dataset-Zenodo%2010048875-brightgreen.svg)](https://doi.org/10.5281/zenodo.10048875)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research code and analysis pipelines for the peer-reviewed paper:

> **Retrieving snow depth distribution by downscaling ERA5 Reanalysis with ICESat-2 laser altimetry**  
> Zhihao Liu, Simon Filhol, Désirée Treichler  
> *Cold Regions Science and Technology*, 2025, 239, 104580.  
> DOI: [10.1016/j.coldregions.2025.104580](https://doi.org/10.1016/j.coldregions.2025.104580) | Open Access: [HAL hal-05443274](https://hal.science/hal-05443274)

---

## 🌟 Gradient Descent Coregistration (GDC) Standalone Library

> [!IMPORTANT]
> The point-to-surface DEM coregistration method originally developed within `xsnow/` has been decoupled, refined, and extracted into a standalone open-source library:
>
> 👉 **[Gradient Descent Coregistration (GDC)](https://github.com/liuh886/gradient-descent-coregistration)** (`import gdc`)
>
> If you are looking for:
> - DEM coregistration with sparse trusted reference points (e.g. ICESat-2, GNSS),
> - Automated DEM Quality Control (QC) reporting with confidence tiers,
> - Lightweight core library with adapters for GeoPandas, Rasterio, and Xarray,
>
> **Please use the standalone repository: [`liuh886/gradient-descent-coregistration`](https://github.com/liuh886/gradient-descent-coregistration)**.  
> The `xsnow` package in this repository remains preserved for exact scientific reproducibility of the paper's original experiments.

---

## 📁 Repository Structure

The repository is organized as follows:

```
subgrid_snow/
├── data/                      # In-situ ground truth validation data
│   └── df_nve_08.csv          # NVE snow observation datasets
├── notebooks/                 # Experiment notebooks and model prediction artifacts
│   ├── data_processing.ipynb  # ICESat-2 ATL08 extraction, DEM sampling, and snow depth retrieval
│   ├── model_prediction.ipynb # Gradient boosted regression (XGBoost) downscaling workflows
│   ├── *.json                 # Quantitative model evaluation metrics across DEM targets
│   └── *.jpg / *.png          # Validation figures, weather scatter charts, and SHAP summary plots
├── results/                   # Regression output plots and scatter figures
│   ├── snowdepth_senoreg_scatter_08_case6_10_corr.jpg
│   └── snowdepth_senoreg_scatter_08_case6_10_original.jpg
├── xsnow/                     # Research toolbox package supporting the paper workflows
│   ├── godh.py                # Point-to-DEM elevation difference sampling and coregistration prototype
│   ├── misc.py                # Coordinate transforms (xy2ij/ij2xy), interpolation, and footprint utilities
│   ├── goregression.py        # Model training, spatial cross-validation, and regression metrics
│   ├── goplot.py              # Geospatial plotting and residual distribution visualizations
│   ├── read_frost.py          # Client API reader for MET Norway Frost weather stations
│   ├── noiseopt.py            # Compass pattern search optimization module
│   ├── pykliep.py             # Density ratio estimation for covariate shift adaptation
│   └── example.py / test_.py  # Internal usage examples and verification scripts
├── requirements.txt           # Python dependency specifications
├── setup.py                   # Package setup configuration for xsnow
└── LICENSE                    # MIT License
```

---

## 🚀 Usage & Reproducibility

### 1. Environment Setup

Clone this repository and install in editable mode:

```bash
git clone https://github.com/liuh886/subgrid_snow.git
cd subgrid_snow

# Option A: Local environment
pip install -e .

# Option B: Run notebooks directly
# Ensure numpy, scipy, pandas, geopandas, rasterio, xdem, and xgboost are installed.
```

### 2. Running Paper Pipelines

- **Data Processing**: Open [`notebooks/data_processing.ipynb`](notebooks/data_processing.ipynb) to inspect the ICESat-2 track extraction, vertical datum transformations (NN2000), and point-to-DEM height sampling.
- **Model Downscaling**: Open [`notebooks/model_prediction.ipynb`](notebooks/model_prediction.ipynb) to run the subgrid snow depth regression, SHAP interpretability analysis, and validation against NVE ground measurements.

---

## 📊 Open Data

The complete multi-DEM dataset (ICESat-2 ATL08 snow-free and snow-covered tracks matched with Norway DTM1, DTM10, Copernicus GLO-30, and FABDEM) is openly available on Zenodo:

- **Zenodo Repository**: [10.5281/zenodo.10048875](https://doi.org/10.5281/zenodo.10048875)

---

## 📖 Citation

If you use code, data, or methods from this repository, please cite:

```bibtex
@article{liu2025retrieving,
  title={Retrieving snow depth distribution by downscaling ERA5 Reanalysis with ICESat-2 laser altimetry},
  author={Liu, Zhihao and Filhol, Simon and Treichler, D{\'e}sir{\'e}e},
  journal={Cold Regions Science and Technology},
  volume={239},
  pages={104580},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.coldregions.2025.104580}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
