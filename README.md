# Shopian Apple Crop Intelligence Dashboard

A minimal, modern Streamlit dashboard for a B.Tech research prototype focused on Shopian, Kashmir.

## Pages
- Overview: historical yield KPIs and trend
- Yield forecast: saved Model 1 estimator, predictors, production estimate, LOYO charts
- Scab weather risk: weather CSV-based provisional LOW/MEDIUM/HIGH heuristic
- Leaf diagnosis: upload a trained Keras model and image to infer class output scores
- Data & methodology: project assumptions and limitations

## Run locally (Windows)
1. Extract the ZIP.
2. Open the extracted `Shopian_Dashboard` folder.
3. Click the address bar in File Explorer, type `cmd`, press Enter.
4. Run:

```cmd
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If Python is installed via launcher, use `py -m pip ...` and `py -m streamlit ...`.

## Model artifacts
- Model 1 artifact is included at `models/model1_yield.joblib`.
- Model 2 is a transparent weather heuristic; no trained binary is required.
- Model 3 trained `.keras` or `.h5` artifact is not included by default; upload it on the Leaf Diagnosis page. Install TensorFlow to enable inference.

## Important limitations
Model 2 estimates weather suitability, not confirmed infection. Leaf wetness is not directly measured and field outbreak labels are absent. Model 1 outputs are estimates and should be interpreted within training-data ranges. Model 3 image output is not a lab diagnosis.

## Deployment
For Streamlit Community Cloud, push the project to a GitHub repository and deploy `app.py`. Keep large/private model files outside public GitHub if needed; configure their availability in the deployed app. TensorFlow may increase deployment size and memory needs.
