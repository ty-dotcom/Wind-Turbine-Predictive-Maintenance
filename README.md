


# Project Design & Implementation Document

## Demo

https://github.com/user-attachments/assets/0633d077-252e-402d-b07a-fae0305056e4


## Streamlit Community Link
https://wind-turbine-predictive-maintenance.streamlit.app

## 1. Project Background & Motivation
With recent events causing limiting our global energy sector, there is a strong emphasis on Renewable energy.
My aim was to develop a simple Streamlit system to handle predictions for anomaly detections for the gearbox in Wind Turbines, using three different classification models and selecting the best model based on the highest f1 score. 

## 2. System Architecture Design

*   **Streamlit Frontend:** A Single Page Application built with `Streamlit`, allowing for CSV/Excel inputs or manual testing using sliders.
*   **Model Management:** Caches the loaded data and model for continuous usage.
*   **Classifier:** A class used to fit and evaluate all models (Isolation Forest, XGBoost Classifier, LGBM Classifier) simultaneously.

<img width="3715" height="2329" alt="Architecture_Wind_Tubine" src="https://github.com/user-attachments/assets/299c4b4d-32d2-4cfb-b44b-c9996abab549" />


## 3. Tech Stack
* Streamlit — Frontend + User Interaction
* scikit-learn — IsolationForest
* XGBoost — XGBClassifier
* LGBModel - LGBMClassifier
* pandas — Data handling

## 5. Model Evaluations

### Methodology
*  **IsolationForest** — Uses the dataset's anomaly rate as
  contamination. Since this model uses -1 and 1 for its classification I mapped its value to 1 and 0 respectively, wherein 1 is an anomaly and 0 is a normal state.

*  **XGBClassifier** — Trained on the labeled training split with
  stratification.

* **LGBMClassifier** — Trained using 31 leaves and a learning rate of 0.01.

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

<img width="536" height="411" alt="image" src="https://github.com/user-attachments/assets/688a55c5-fe72-4705-9ee8-f86ea991d736" />

The SCADA dataset had 97.1% cases being normal and only 2.1% being labeled as anomalies, showing high class imbalance.

<img width="1189" height="495" alt="image" src="https://github.com/user-attachments/assets/ffa31af2-cbd7-4b31-9b91-8131bcbfdeb7" />

The two most used features for these models are vibration_x and oil_pressure, as they provided the most information in classifying the gearbox states.

XGBClassifier delegated its importance to the vibration_x metric alone holding (87% gain). While LightGBM split its importance
among three metrics vibration_x, oil_pressure, and vibration_y which all resulted in a 90% gain.

### Model Performances
- Isolation Forest Performance
<img width="501" height="192" alt="image" src="https://github.com/user-attachments/assets/db1d58da-1d2a-4de1-af10-304abb6cc178" />

- LightGBM Classifier Performance
<img width="487" height="191" alt="image" src="https://github.com/user-attachments/assets/72fb5f2f-c4df-4d5e-ac5f-3cda5cc74126" />

- XGBoost Classifier Performance
<img width="483" height="192" alt="image" src="https://github.com/user-attachments/assets/23e4ee76-ce81-4e74-b964-2408e936efa4" />


<img width="1488" height="494" alt="image" src="https://github.com/user-attachments/assets/a67895c4-53c9-4a0d-b943-9977cff3f11e" />

The Isolation Forest produced 23 False Positives and 24 False Negatives, whilst the other two models performed perfectly.

<img width="1489" height="578" alt="image" src="https://github.com/user-attachments/assets/1fe1688e-1a5f-4c83-98f9-48678003135a" />

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
