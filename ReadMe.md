# Hybrid Real Estate Price App (LGBM ↔ RF)

Streamlit App (Use this link to try the demo): https://idx-exchange-ca-housing-price-emutxgkxnyargbdvmkq6e7.streamlit.app/

## Purpose
Single‐residence housing price prediction in California, using **hybrid models**:
- **Auto**: use **RF** for *Los Angeles* & *Riverside*, **LGBM** elsewhere  
- **Manual override**: *Force LGBM* / *Force RF*

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py

## File Structure
Hybrid_App/
├─ app.py
├─ preprocessing.py              # compatibility shim → re-exports from rf/RF_preprocessing.py
├─ rf/
│  ├─ __init__.py
│  └─ RF_preprocessing.py        # RF feature engineering / transformers
├─ lgbm/
│  ├─ __init__.py
│  └─ LGBM_preprocessing.py      # preprocess_input(df, ref_values)
└─ model/
   ├─ RF_model.pkl
   ├─ RF_preprocessor.pkl
   ├─ RF_price_bounds.pkl      
   └─ LGBM_model.pkl

## Disclaimer
This is just a demo app for exploring hybrid models. It can make mistakes :)

## Credits
Built by Group DS26 during the IDX Exchange internship, mentored by Aidan
