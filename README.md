## Thesis Code

This folder (which is submitted with the thesis) has the codes used for the data analysis for my master's thesis Predictive and Unsupervised Machine Learning of Postprandial Glucose Dynamics from Multimodal Data Combining Meal Images and Continuous Glucose Monitoring.

Data and intermediate results are excluded.

### Note on data
Since raw data and intermediate results are not included, the pipeline cannot be re-run directly. 

Code here is to show how the analysis was implemented.

### Setup
conda env create -f environment.yml <br>
conda activate circadian

### Structure

functions/- utils.py: all the functions that are later used in the notebooks, <br>
            config.py:has the data paths (hardcoded for my local setup), imports and some constants used, <br>
            color_palettes.py


notebooks/- full analysis pipeline notebooks, order:
  1. metadata_visualisation.ipynb
  2. geminioutput_decode.ipynb
  3. cgm_data_feature_extraction.ipynb
  4. fasting_times.ipynb
  5. glucose_profiles_and_circadian_baseline_visuals.ipynb
  6. meal_overlapped_cgm_plot.ipynb
  7. meal_cgm_pairing_and_recording_rate.ipynb
  8. clustering.ipynb
  9. ppgr_iauc_analysis_data_prep.ipynb
  10. iauc_prediction_models.ipynb
  11. time_series_data_preparation.ipynb
  12. timeseries_prediction_models.ipynb


