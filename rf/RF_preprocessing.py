from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

class RealEstatePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        final_features,
        drop_high_missing=True,
        enforce_property_type=True,
        set_index=True,
        high_missing_threshold=0.5
    ):
        self.final_features = final_features
        self.drop_high_missing = drop_high_missing
        self.enforce_property_type = enforce_property_type
        self.set_index = set_index
        self.high_missing_threshold = high_missing_threshold

        # For saving preprocessing state
        self.median_values = {}
        self.freq_maps = {}
        self.cap_values = {}
        self.cols_to_drop = []
        self.kmeans_model = None  # KMeans model for clustering

    def clean_levels(self, val):
        if pd.isna(val):
            return np.nan
        levels = str(val).split(',')
        levels_set = set(levels)
        if levels_set in [{'One', 'Two', 'ThreeOrMore', 'MultiSplit'}, {'MultiSplit'}]:
            return np.nan
        score = 0
        for level in levels:
            if 'Three' in level:
                score = max(score, 3)
            elif 'Two' in level:
                score = max(score, 2)
            elif 'One' in level:
                score = max(score, 1)
        return score if score > 0 else np.nan

    def create_latlon_cluster(self, df, n_clusters=10, fit=False):
        if fit:
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df["LatLonCluster"] = self.kmeans_model.fit_predict(df[["Latitude", "Longitude"]].fillna(0))
        else:
            if self.kmeans_model is None:
                raise ValueError("KMeans model has not been fitted. Please run fit() first.")
            df["LatLonCluster"] = self.kmeans_model.predict(df[["Latitude", "Longitude"]].fillna(0))

    def create_latlon_bins(self, df, lat_bin_size=0.1, lon_bin_size=0.1):
        df["LatBin"] = (df["Latitude"] // lat_bin_size) * lat_bin_size
        df["LonBin"] = (df["Longitude"] // lon_bin_size) * lon_bin_size

    def _normalize_categories(self, df):
        for col in ["City", "CountyOrParish", "HighSchoolDistrict"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()

    def fit(self, X, y=None):
        df = X.copy()

        if self.enforce_property_type:
            df = df[(df["PropertyType"] == "Residential") & (df["PropertySubType"] == "SingleFamilyResidence")]

        if self.drop_high_missing:
            missing_rate = df.isna().mean()
            self.cols_to_drop = missing_rate[missing_rate > self.high_missing_threshold].index.tolist()
            df.drop(columns=self.cols_to_drop, inplace=True)

        self._normalize_categories(df)

        for col in df.select_dtypes(include=[np.number]):
            self.median_values[col] = df[col].median()

        for col in ['City', 'PostalCode', 'CountyOrParish', 'HighSchoolDistrict']:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True)
                self.freq_maps[col] = freq.to_dict()

        cap_cols = ['LivingArea', 'LotSizeSquareFeet', 'BathroomsTotalInteger', 'BedroomsTotal', 'GarageSpaces', 'AssociationFee']
        for col in cap_cols:
            if col in df.columns:
                self.cap_values[col] = df[col].quantile(0.99)

        if "Latitude" in df.columns and "Longitude" in df.columns:
            df[["Latitude", "Longitude"]] = df[["Latitude", "Longitude"]].fillna(df[["Latitude", "Longitude"]].median())
            self.create_latlon_cluster(df, fit=True)

        return self

    def transform(self, X):
        df = X.copy()

        if self.set_index and "ListingKey" in df.columns:
            df.set_index("ListingKey", inplace=True)

        if self.enforce_property_type:
            df = df[(df["PropertyType"] == "Residential") & (df["PropertySubType"] == "SingleFamilyResidence")]

        df.drop(columns=[c for c in self.cols_to_drop if c in df.columns], errors="ignore", inplace=True)

        self._normalize_categories(df)

        if "Levels" in df.columns:
            df["Levels"] = df["Levels"].apply(self.clean_levels).fillna(1).astype(int)

        if "Flooring" in df.columns:
            flooring_filled = df["Flooring"].fillna("").astype(str)
            for mat in ['Carpet', 'Tile', 'Wood', 'Laminate', 'Vinyl', 'Stone', 'Concrete', 'Bamboo']:
                df[f"Has{mat}"] = flooring_filled.str.contains(mat, case=False).astype(int)
            df["FlooringMissing"] = df["Flooring"].isna().astype(int)
            df.drop(columns="Flooring", inplace=True)

        if "Latitude" in df.columns and "Longitude" in df.columns:
            df[["Latitude", "Longitude"]] = df[["Latitude", "Longitude"]].fillna(df[["Latitude", "Longitude"]].median())
            self.create_latlon_cluster(df, fit=False)
            self.create_latlon_bins(df)
            df.drop(columns=["Latitude", "Longitude"], inplace=True)

        for col, fmap in self.freq_maps.items():
            df[f"{col}Freq"] = df[col].map(fmap).fillna(0)

        for col, med in self.median_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        for col, cap in self.cap_values.items():
            if col in df.columns:
                df[col] = np.where(df[col] > cap, cap, df[col])

        if "ParkingTotal" in df.columns:
            df["ParkingTotal"] = df["ParkingTotal"].mask(df["ParkingTotal"] < 0, np.nan).fillna(0)
            df["ParkingTotal"] = np.minimum(df["ParkingTotal"], 50)

        if "MainLevelBedrooms" in df.columns:
            df["MainLevelBedrooms"] = np.minimum(df["MainLevelBedrooms"], 10)

        if "YearBuilt" in df.columns:
            df["YearBuilt"] = np.minimum(df["YearBuilt"], 2024)

        if "AssociationFee" in df.columns:
            df["AssociationFee"] = np.minimum(df["AssociationFee"], 2000)

        df_final = df[self.final_features].copy()
        if "ClosePrice" in df.columns:
            df_final["ClosePrice"] = df["ClosePrice"]
        df_final.dropna(inplace=True)

        return df_final
