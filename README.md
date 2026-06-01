# BharatNXT — SME FinTech Analytics Dashboard

Data-driven customer segmentation, classification, association rule mining, regression and new-customer scoring for BharatNXT — a B2B credit payment platform for Indian SMEs.

## Deploy on Streamlit Cloud

1. Fork or upload this repo to GitHub (all files in root — no subfolders except `pages/`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
├── app.py                        # Main Streamlit entry point
├── utils.py                      # Shared data loading, preprocessing, model training
├── generate_data.py              # Run once to regenerate the synthetic dataset
├── bharatnxt_survey_data.csv     # 2000-row synthetic survey dataset
├── requirements.txt
├── README.md
└── pages/
    ├── __init__.py
    ├── page_overview.py          # Page 0: Overview and data health
    ├── page_descriptive.py       # Page 1: Descriptive analytics
    ├── page_diagnostic.py        # Page 2: Diagnostic analytics + chi-square
    ├── page_clustering.py        # Page 3: K-Means clustering
    ├── page_arm.py               # Page 4: Association rule mining (Apriori)
    ├── page_classification.py    # Page 5: RF + LR, accuracy/precision/recall/F1/ROC
    ├── page_regression.py        # Page 6: Linear regression, GTV prediction
    └── page_scoring.py           # Page 7: New customer CSV upload + scoring engine
```

## Analytics layers

| Layer | Page | Algorithms |
|-------|------|-----------|
| Descriptive | 0, 1 | Statistics, histograms, heatmaps, funnels |
| Diagnostic | 2 | Correlation, chi-square, barrier analysis |
| Predictive | 3, 5, 6 | K-Means, Random Forest, Logistic Regression, Linear Regression |
| Prescriptive | 4, 7 | Apriori/ARM, rule-based action engine |

## Dataset

2,000 synthetic Indian SME survey respondents across 6 personas:
- P1: Tech-savvy IT founders (700 rows)
- P2: Established RE brokers (500 rows)
- P3: Traditional SMEs (300 rows)
- P4: Young consultants (250 rows)
- P5: Fraud-burnt skeptics (150 rows)
- P6: High-volume power users (100 rows)

Target: `adoption_intent` (0=Definitely not, 1=Probably not, 2=Probably yes, 3=Definitely yes)
