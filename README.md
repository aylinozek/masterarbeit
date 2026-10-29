## Master's Thesis Code

This folder (which is submitted with the thesis) has the codes used for the data analysis for my master's thesis Predictive and Unsupervised Machine Learning of Postprandial Glucose Dynamics from Multimodal Data Combining Meal Images and Continuous Glucose Monitoring.

Data and intermediate results are excluded.

### Note on data
Since raw data and intermediate results are not included, the pipeline cannot be re-run directly. 

Code here is to show how the analysis was implemented.

## Objective
The objective of this thesis project is to create a machine learning based data analysis pipeline
to analyze multimodal data to predict/replicate glucose responses using the meal photos recorded by
participants, to identify the most important features of a meal for this prediction task and to discover
hidden patterns. The primary aim is evaluate a personalized approach through residual correction
against the population level global model. In addition to this primary direction, the study also aims to
asses the contribution of different feature channels to PPGR prediction, i.e.meal events (LLM extracted
content information and timing), demographics and baseline characteristics, CGM extracted summary
features and CGM derived circadian features. Before conducting the main predictive analysis, some
explatory analysis aims are to predict the possible meal events from CGM data to estimate meal recording
rates and to explore eating habits and response clusters.

### Data
The data used is the self-reported and non- standardized meals recorded by the participants, 
the 10-day CGM recordings which act as both the target variable (PPGR) and glucose metabolism 
summary features extracted as the mean, standard deviation, time in range (TIR), speed, acceleration and 
cosinor analysis outputs representing the circadian behaviour.
A notable limitation of this study design is that the self-reported meals can be very noisy and incomplete.
This limitation was addressed in the recording rate analysis before moving on to the prediction models.

### Results
In the residual correction fusion approach a ridge regression was fitted on the errors between the global
model predictions on each participant’s after meal glucose responses and the true responses. A corrector
blending coefficient was calculated for each participant, representing the deviation of that participant’s
trend from the global model. This method was applied on the best performing model + feature subset
pair for both the iAUC and time series predicitons which were XGBoost with the full feature subset
(Meal + Meta + Glucose + Circadian) and LSTM with Glucose + Meal Time feature subset, respectively.
Residual correction fusion models achieved a mean RMSE of 1.187± 0.29, with a minimum of 0.685 and
a maximum of 1.932 for the scalar prediction and a mean RMSE of 0.932± 0.28, with a minimum of
0.422 and a maximum of 1.978 for the time series prediction. These results showed an improvement on
the global models, especially for the iAUC prediction.
