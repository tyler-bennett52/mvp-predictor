# MLB MVP Predictor

A machine learning system that predicts MLB Most Valuable Player award winners with 96% historical accuracy.

## Live Demo

[View the app on Streamlit Cloud](https://mvp-predictor-bennett.streamlit.app/)

## Features

- Predict MVP winners using 55 statistical features
- Explore historical MVP races from 1945-2024
- Analyze controversial selections with the "Robbery Index"
- Make custom predictions with your own statistics

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

The system uses a pairwise learning-to-rank approach with gradient boosting. It compares candidates head-to-head and selects winners through tournament elimination.

- 96% accuracy on historical data (145/151 correct)
- 55 features including WAR, team wins, and narrative factors
- Trained on MVP voting patterns from 1945-2024
