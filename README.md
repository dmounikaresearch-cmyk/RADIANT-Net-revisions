# RADIANT-Net revisited: data and code

Data and code for the revised manuscript

> **RADIANT-Net revisited: a leakage-aware evaluation of multimodal deep learning for natural hazard type classification from environmental and catalogue data**
> Dannuri Mounika, Johnson Kolluri, Sunil Srinivas, Kiran Siripuri. Submitted to *Environmental Sciences Europe*.

The repository contains 500 catalogued natural-hazard events (100 each of cyclone, earthquake, flood, landslide and wildfire). Each event has two feature sets:

- the **original** 15 features used in the first submission, audited in Sect. 2.2 of the paper;
- **re-extracted** features from ERA5 reanalysis and the Copernicus GLO-90 elevation model, which are the basis of the main analysis.

The repository also contains the scripts that reproduce every table and figure.

## Contents

| Path | Description |
|---|---|
| `data/radiant_real_disaster_data.csv` | Original dataset (15 features), as produced by `code/original_project/fetch_real_data.py` |
| `data/radiant_clean_openmeteo_data.csv` | Re-extracted features for the same 500 events, produced by `code/revision/fetch_clean_features.py` |
| `data/radiant_clean_openmeteo_data_log.txt` | Extraction settings, missing-value counts and summary statistics of the re-extraction |
| `data/events_clean_with_regions.csv` | Re-extracted features with event key and leave-region-out region (R0–R7); input of the main analysis |
| `data/events_original_with_regions.csv` | Original features with event key and region. Surface pressures above 1,085 hPa are set to missing (Sect. 2.2). |
| `code/original_project/` | The authors' original prototype: data-assembly script (`fetch_real_data.py`), model (`model.py`, 45,189 parameters with 19 inputs), training script and configuration |
| `code/revision/fetch_clean_features.py` | Re-extracts all features from the Open-Meteo archive (ERA5, ERA5-Land) and elevation service (Copernicus GLO-90). It adds no noise, uses no fallback values and uses no label information. |
| `code/revision/run_revision_experiments.py` | All reviewer-requested experiments: repeated, event-grouped and leave-region-out validation, eight baselines, paired tests, ablation, SHAP, uncertainty, out-of-distribution detection, stress tests, timing |
| `code/revision/analysis3.py` | Pooled per-class results, class-prior reweighting and the manuscript figures |
| `code/revision/groups*.json` | Feature-group definitions for the ablation and missing-modality analyses |
| `supplementary/` | Supplementary Tables S1 (event list) and S2 (feature statistics) and Figs. S1 (feature distributions) and S2 (learning curve) |

## Feature provenance (summary)

**Original features** (Table 3a of the paper):

- *Weather and soil:* Open-Meteo values with 8% multiplicative noise added.
- *River level, vegetation dryness and water index:* formulas of those values.
- *Slope:* a latitude rule, with about 8° added for landslides.
- *Seismic activity and ground displacement:* catalogue magnitude for earthquakes, and class-specific random values for other hazards.
- *Cyclone wind speed:* GDACS storm severity.
- *Population density, road accessibility and risk score:* the catalogue alert score plus noise.

These features are used only to quantify leakage.

**Re-extracted features** (Table 3b): ERA5 means over the event date −2 to +1 days of the following variables:

- precipitation;
- 2-m temperature;
- relative humidity;
- daily maximum 10-m wind;
- surface pressure;
- FAO reference evapotranspiration.

In addition, there are 30-day antecedent precipitation, ERA5-Land soil moisture (0–7 cm), elevation, and slope computed from Copernicus GLO-90 by central differences.

## Reproducing the results

Requirements: Python 3.11 or later, with `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `shap` and `matplotlib`.

Run from the repository root:

```bash
# 1. (optional) re-extract the features from data/radiant_real_disaster_data.csv; needs internet, about 20 minutes
python code/revision/fetch_clean_features.py

# 2. main analysis: 10 re-extracted features
python code/revision/run_revision_experiments.py --data data/events_clean_with_regions.csv \
  --label disaster_type --event-col event_id --region-col region --date-col event_date \
  --features precip_mean_mm_day precip_antecedent_30d_mm temperature_mean_c rel_humidity_mean_pct \
             wind_max_mean_kmh surface_pressure_mean_hpa et0_mean_mm_day soil_moisture_0_7cm_m3m3 \
             elevation_m slope_deg \
  --groups code/revision/groups_clean.json --seeds 10 --region-folds 8 --out results_clean

# 3. leakage analysis: 15 original features
python code/revision/run_revision_experiments.py --data data/events_original_with_regions.csv \
  --label disaster_type --event-col event_id --region-col region --date-col event_date \
  --features rainfall_mm temperature_c humidity_percent wind_speed_kmh atmospheric_pressure_hpa \
             river_level_m soil_moisture_percent slope_degrees vegetation_dryness seismic_activity \
             ground_displacement_mm satellite_water_index population_density road_accessibility \
             historical_risk_score \
  --groups code/revision/groups.json --seeds 10 --region-folds 8 --out results15_orig

# 4. pooled per-class results, priors and figures (written to analysis3/)
python code/revision/analysis3.py
```

Steps 2 and 3 take one to two hours each on a laptop CPU. Results are saved after every split, so an interrupted run can simply be restarted.

## Data sources

- GDACS: https://www.gdacs.org
- USGS ComCat: https://earthquake.usgs.gov/fdsnws/event/1/
- NOAA Storm Events Database: https://www.ncdc.noaa.gov/stormevents/
- Open-Meteo historical weather and elevation APIs: https://open-meteo.com (ERA5/ERA5-Land, Copernicus GLO-90)

Please cite the original data providers as well as the paper when using these data.
