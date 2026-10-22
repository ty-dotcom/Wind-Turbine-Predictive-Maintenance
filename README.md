

# Project Design & Implementation Document

## Demo

## 1. Project Background & Motivation
With recent events causing limiting our global energy sector, there is a strong emphasis on Renewable energy.
My aim was to develop a simple Streamlit system to handle predictions for anomaly detections for the gearbox in Wind Turbines, using three different classification models and selecting the best model based on the highest f1 score. 

## 2. System Architecture Design

*   **Streamlit Frontend:** A Single Page Application built with `Streamlit`, allowing for CSV/Excel inputs or manual testing using sliders.
*   **Model Management:** Caches the loaded data and model for continuous usage.
*   **Classifier:** A class used to fit and evaluate all models (Isolation Forest, XGBoost Classifier, LGBM Classifier) simultaneously.

## 3. Tech Stack
* Streamlit — Frontend + User Interaction
* scikit-learn — IsolationForest
* XGBoost — XGBClassifier
* LGBModel - LGBMClassifier
* pandas — Data handling

## 5. Model Evaluations

### Methodology
*  **IsolationForest** — Unsupervised Model. Uses the dataset's anomaly rate as
  contamination. Since this model uses -1 and 1 for its classification I mapped its value to 1 and 0 respectively, wherein 1 is an anomaly and 0 is a normal state.

*  **XGBClassifier** — Supervised Model. Trained on the labeled training split with
  stratification.

* **LGBMClassifier** — Supervised Model. Trained using 31 leaves and a learning rate of 0.01.

Once each model is trained it is then evaluated on the test set using weighted
precision, recall, and f1. The best model is selected based on the highest f1 score. 
Though users can choose to switch between models.

### Input Features
| Feature | Range | Unit |
|---|-------|---|
| gearbox_oil_temp | 40–100 | °C |
| gearbox_bearing_temp | 50–100 | °C |
| vibration_x | 0–0.05 | g/m/s² |
| vibration_y | 0–0.04 | g/m/s² |
| vibration_z | 0–0.06 | g/m/s² |
| oil_pressure | 3–5   | bar |
| particle_count | 100–2000 | count/ml |

### Observations
| Observation                                                                                                                                                                   | Implication                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| vibration_x, vibration_y, vibration_z are perfectly correlated                                                                                                                | 	Although the models recieve three separate vibration metrics, due to their perfect correlation it is not treated as three independent metrics. In the XGBClassifier only one vibration metric is used when determining an anomaly, while the LGBMClassifier utilized two. |
| Vibration correlates with is_anomaly at r = 0.72                                                                                                                              | 	A strong correlation exists between the vibration metrics and a state being labeled as an anomaly, which explains why the models focused heavily on the vibration metrics.                                                                                                |
| particle_count has a strong negative-correlation with gearbox_oil_temp (r = -0.72) and gearbox_bearing_temp (r = -0.69) and a weak negative correlation to is_anomaly (-0.05) | The particle count metric contributes no unique information for the classifier.                                                                                                                                                                                            |
| oil_pressure has a fairly strong negative correlation with is_anomaly (r = -0.57)                                                                                             | 	This correlation was seen being used heavily by the LGBMClassifier in its predictions, and only fairly used for the XGBClassifier.                                                                                                                                        |


The SCADA dataset had 97.1% cases being normal and only 2.1% being labeled as anomalies, showing high class imbalance.

The two most used features for these models are vibration_x and oil_pressure, as they provided the most information in classifying the gearbox states.

XGBClassifier delegated its importance to the vibration_x metric alone holding (87% gain). While LightGBM split its importance
among three metrics vibration_x, oil_pressure, and vibration_y which all resulted in a 90% gain.

The Isolation Forest produced 23 False Positives and 24 False Negatives, whilst the other two models performed perfectly.

Using a Precision Recall Curve, each model achieved an AP = 1.00 across all thresholds.

### Colab used to evaluate model
https://colab.research.google.com/drive/1T4rvQJnoPi5mkgK6j9wwV3VRm3UX7om3?usp=sharing

## Run Locally

Clone the project and open the folder

```bash
  git clone https://github.com/ty-dotcom/Wind-Turbine-Predictive-Maintenance.git
  cd Wind-Turbine-Predictive-Maintenance
```

Create and activate a virtual environment

#### Windows 
```bash
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
```

#### Linux & MacOS
```bash
  python -m venv .venv
  source .venv/bin/activate
```

Install requirements

```bash
  pip install -r requirements.txt
```

Start the App 

```bash
  streamlit run streamlit_app.py
```
