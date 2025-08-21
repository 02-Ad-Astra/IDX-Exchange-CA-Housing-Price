# app.py
import os
import sys
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from importlib import import_module

# ==================== Page Setup ====================
st.set_page_config(page_title="Hybrid CA House Price App", page_icon="🏠", layout="wide")
st.title("🏠 Hybrid California House Price Predictor (LGBM ↔ RF)")

# ==================== Paths & Imports (match your tree) ====================
# All pkl files live under ./model
MODELS_DIR = "model"

LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "LGBM_model.pkl")
RF_MODEL_PATH   = os.path.join(MODELS_DIR, "RF_model.pkl")
RF_PREPROC_PATH = os.path.join(MODELS_DIR, "RF_preprocessor.pkl")
RF_BOUNDS_PATH  = os.path.join(MODELS_DIR, "RF_price_bounds.pkl")  # optional

# LGBM preprocessing lives at lgbm/LGBM_preprocessing.py
# must define: preprocess_input(df, ref_values)
LGBM_PREPROCESSING_MODULE = "lgbm.LGBM_preprocessing"

# (Optional) sample CSVs to build ref_values for LGBM; keep None if you don't use it
LGBM_DATA_DIR = "data"

# Make sure package imports (rf, lgbm) work no matter where streamlit is launched
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ==================== Routing Rules ====================
AUTO_RF_COUNTIES = {"Los Angeles", "Riverside"}  # RF for these; LGBM otherwise

def normalize_county_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"county$", "", s).strip()
    if s in {"la", "l.a.", "los angeles", "los-angeles"}:
        return "Los Angeles"
    if s in {"riverside"}:
        return "Riverside"
    return " ".join(w.capitalize() for w in s.split())

def route_model(county: str, override: str) -> str:
    if override == "Force LGBM":
        return "LGBM"
    if override == "Force RF":
        return "RF"
    return "RF" if normalize_county_name(county) in AUTO_RF_COUNTIES else "LGBM"

# ==================== Cached Artifacts ====================
@st.cache_resource(show_spinner=False)
def load_lgbm_artifacts():
    if not os.path.exists(LGBM_MODEL_PATH):
        raise FileNotFoundError(f"Missing file: {LGBM_MODEL_PATH}")
    model = joblib.load(LGBM_MODEL_PATH)

    try:
        pre_mod = import_module(LGBM_PREPROCESSING_MODULE)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import '{LGBM_PREPROCESSING_MODULE}'. "
            "Ensure lgbm/LGBM_preprocessing.py exists and defines preprocess_input(df, ref_values)."
        ) from e
    if not hasattr(pre_mod, "preprocess_input"):
        raise AttributeError("LGBM_preprocessing.py must define: preprocess_input(df, ref_values)")
    preprocess_input = getattr(pre_mod, "preprocess_input")

    @st.cache_data(show_spinner=False)
    def load_combined_data(folder=LGBM_DATA_DIR):
        if not folder or not os.path.isdir(folder):
            # Fallback defaults if no CSVs provided
            return pd.DataFrame({
                "LivingArea":[1500], "LotSizeAcres":[0.25], "YearBuilt":[1990],
                "City":["San Diego"], "PostalCode":["92101"], "PropertyType":["Residential"]
            })
        dfs = []
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(folder, f)))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame({
            "LivingArea":[1500], "LotSizeAcres":[0.25], "YearBuilt":[1990],
            "City":["San Diego"], "PostalCode":["92101"], "PropertyType":["Residential"]
        })

    sample_df = load_combined_data(LGBM_DATA_DIR)
    ref_values = {
        "LivingArea": sample_df["LivingArea"].median(),
        "LotSizeAcres": sample_df["LotSizeAcres"].median(),
        "HomeAge": 2025 - sample_df["YearBuilt"].median(),
        "City": sample_df["City"].mode()[0],
        "PostalCode": sample_df["PostalCode"].mode()[0],
        "PropertyType": sample_df["PropertyType"].mode()[0]
    }
    return {"model": model, "preprocess_input": preprocess_input, "ref_values": ref_values}

@st.cache_resource(show_spinner=False)
def load_rf_artifacts():
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(f"Missing file: {RF_MODEL_PATH}")
    if not os.path.exists(RF_PREPROC_PATH):
        raise FileNotFoundError(f"Missing file: {RF_PREPROC_PATH}")

    # Safety shim in case old pickles import 'preprocessing'
    # (you also have root-level preprocessing.py re-exporting from rf.RF_preprocessing)
    try:
        import rf.RF_preprocessing as _rfprep
        sys.modules.setdefault("preprocessing", _rfprep)
    except Exception:
        pass

    model = joblib.load(RF_MODEL_PATH)
    preprocessor = joblib.load(RF_PREPROC_PATH)

    bounds = None
    if os.path.exists(RF_BOUNDS_PATH):
        try:
            bounds = joblib.load(RF_BOUNDS_PATH)
        except Exception:
            bounds = None
    return {"model": model, "preprocessor": preprocessor, "bounds": bounds}

# Load once (cached)
lgbm_art, rf_art = None, None
try:
    lgbm_art = load_lgbm_artifacts()
except Exception as e:
    st.error(f"Failed to load LGBM artifacts: {e}")
try:
    rf_art = load_rf_artifacts()
except Exception as e:
    st.error(f"Failed to load RF artifacts: {e}")

# ==================== Sidebar Controls ====================
with st.sidebar:
    st.header("⚙️ Settings")
    override = st.radio("Model Mode", options=["Auto", "Force LGBM", "Force RF"], index=0)
    county   = st.text_input("County (used for Auto routing)", value="Los Angeles")
    chosen   = route_model(county, override)
    st.markdown(f"**Current model:** `{chosen}`")

# ==================== LGBM Form ====================
def lgbm_form():
    st.subheader("🧾 LGBM: Enter Property Details")
    with st.form("lgbm_form"):
        LivingArea   = st.number_input("Living Area (sq ft)", min_value=200, max_value=10000, value=1500)
        LotSizeAcres = st.number_input("Lot Size (acres)", min_value=0.01, max_value=100.0, value=0.25)
        YearBuilt    = st.number_input("Year Built", min_value=1800, max_value=2025, value=1990)
        PropertyType = st.selectbox("Property Type", [
            'Residential','ResidentialLease','ResidentialIncome','Land','CommercialLease',
            'ManufacturedInPark','CommercialSale','BusinessOpportunity'
        ], index=0)
        City         = st.text_input("City", value="San Diego")
        PostalCode   = st.text_input("Postal Code", value="92101")
        submitted    = st.form_submit_button("Predict Price (LGBM)")

    if submitted:
        if lgbm_art is None:
            st.error("LGBM artifacts are not loaded.")
            return
        try:
            input_df = pd.DataFrame({
                "LivingArea": [LivingArea],
                "LotSizeAcres": [LotSizeAcres],
                "YearBuilt": [YearBuilt],
                "PropertyType": [PropertyType],
                "City": [City],
                "PostalCode": [PostalCode]
            })
            processed = lgbm_art["preprocess_input"](input_df, lgbm_art["ref_values"])
            log_pred  = lgbm_art["model"].predict(processed)[0]
            price     = np.expm1(log_pred)
            st.success(f"💰 Predicted Home Price: **${price:,.0f}**")
        except Exception as e:
            st.error(f"❌ LGBM prediction error: {e}")

# ==================== RF Form ====================
def rf_form():
    st.subheader("🧾 RF: Input the home's features")
    sqft    = st.number_input("Living Area (sqft)", 500.0, 10000.0, value=2000.0)
    baths   = st.number_input("Bathrooms", 0, 10, step=1, value=2)
    beds    = st.number_input("Bedrooms", 0, 10, step=1, value=3)
    lot_sz  = st.number_input("Lot Size (sqft)", 0.0, 50000.0, value=5000.0)
    garage  = st.number_input("Garage Spaces (decimal)", 0.0, 20.0, value=2.0)
    parking = st.number_input("Total Parking Spaces (integer)", 0.0, 50.0, step=1.0, value=2.0)
    year    = st.number_input("Year Built", 1700.0, 2025.0, value=2000.0)
    assoc   = st.number_input("Association Fee ($/mo)", 0.0, 3000.0, value=0.0)
    mainbd  = st.number_input("Main Level Bedrooms", 0.0, 5.0, step=1.0, value=2.0)
    levels  = st.selectbox("Levels", ["One", "Two", "ThreeOrMore", "MultiSplit"])
    flooring = st.multiselect(
        "Flooring Materials",
        ['Carpet','Tile','Wood','Laminate','Vinyl','Stone','Concrete','Bamboo'],
        default=['Wood']
    )
    fireplace = st.checkbox("Fireplace", value=True)
    pool      = st.checkbox("Private Pool", value=False)
    view      = st.checkbox("Scenic View", value=False)

    zipcode = st.text_input("ZIP Code", "90210")
    city    = st.text_input("City", "Beverly Hills")
    county_ = st.text_input("County/Parish", "Los Angeles")
    district= st.text_input("High School District", "Los Angeles Unified")
    lat     = st.number_input("Latitude", 0.0, 120.0, value=34.09)
    lon     = st.number_input("Longitude", -180.0, 0.0, value=-118.41)

    if st.button("Predict Price (RF)"):
        if rf_art is None:
            st.error("RF artifacts are not loaded.")
            return
        try:
            flooring_str = ", ".join(flooring) if flooring else ""
            raw = pd.DataFrame([{
                "LivingArea": sqft,
                "BathroomsTotalInteger": baths,
                "BedroomsTotal": beds,
                "LotSizeSquareFeet": lot_sz,
                "GarageSpaces": garage,
                "ParkingTotal": parking,
                "YearBuilt": year,
                "AssociationFee": assoc,
                "MainLevelBedrooms": mainbd,
                "Levels": levels,
                "FireplaceYN": int(fireplace),
                "PoolPrivateYN": int(pool),
                "ViewYN": int(view),
                "Flooring": flooring_str,
                "PostalCode": zipcode,
                "City": city,
                "CountyOrParish": county_,
                "HighSchoolDistrict": district,
                "Latitude": lat,
                "Longitude": lon,
                "PropertyType": "Residential",
                "PropertySubType": "SingleFamilyResidence"
            }])

            processed = rf_art["preprocessor"].transform(raw)
            X = processed[rf_art["model"].feature_names_in_]
            log_pred = rf_art["model"].predict(X)[0]
            price = np.expm1(log_pred)
            st.success(f"💰 Estimated Home Price: **${price:,.0f}**")
        except KeyError as e:
            st.error(f"Column alignment error (missing column): {e}")
        except Exception as e:
            st.error(f"❌ RF prediction error: {e}")

# ==================== Render ====================
if route_model(county, override) == "LGBM":
    lgbm_form()
else:
    rf_form()

# ==================== Notes ====================
with st.expander("ℹ️ Notes"):
    st.markdown("""
- **Auto mode**: Los Angeles / Riverside → **RF**; all other counties → **LGBM**.  
- Sidebar lets you force a model (useful for debugging or A/B tests).  
- Models were trained on `log1p(target)`, so predictions are converted back with `expm1`.  
""")
