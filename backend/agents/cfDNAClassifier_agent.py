import os
from .base_agent import BaseAgent
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.pipeline import Pipeline
from .cfDNAmeth_pipeline import impute_and_scale, load_preprocessed_methylation_data, preprocessing_methylation_data, preprocessing_clinical_data
from .utils import flatten_dict_values, is_chromosomal, select_random_methylation_row

class cfDNAClassifierAgent(BaseAgent):
    def __init__(self, model_name="combined", model_type="rf", fold=1):
        super().__init__("cfDNAClassifierAgent")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_name = model_name
        self.model_type = model_type
        self.fold = fold

        self.models_dir = os.path.join(base_dir, "models")
        self.features_dir = os.path.join(base_dir, "results", "selected_features")
        self.staging_dir = os.path.join(base_dir, "data", "staging")

        # Load model
        model_path = os.path.join(self.models_dir, f"{self.model_name}_{self.model_type}_fold{self.fold}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)

        self.X_input = []


    def run(self, clinical_df, methyl_df) -> pd.DataFrame:        

        clinical = preprocessing_clinical_data(clinical_df)

        # Load model
        ##model_path = os.path.join(self.models_dir, f"{self.model_name}_{self.model_type}_fold{self.fold}.pkl")
        ##if not os.path.exists(model_path):
        ##    raise FileNotFoundError(f"Model not found: {model_path}")
        #model = joblib.load(model_path)

        # Include all features expected by model, set as 0 if not present
        expected_features = self.model.feature_names_in_
        clinical = clinical.reindex(columns=expected_features, fill_value=0)
        # Remove chromosomal features
        clinical = clinical.loc[:, ~clinical.columns.map(is_chromosomal)]
        

        # Load selected features
        ##selected_features_path = os.path.join(self.features_dir, f"boruta_selected_features_fold{self.fold}.txt")
        ##if os.path.exists(selected_features_path):
        ##    with open(selected_features_path, "r") as f:
        ##        selected_features = [line.strip() for line in f.readlines()]
            #print(f"Selected features loaded for fold {fold}.")
        ##else:
        ##    raise FileNotFoundError(f"Selected features file not found for fold {self.fold}: {selected_features_path}")

        # Filter methylation data based on selected features
        ##X_meth_sel = self.methylation[selected_features]
        X_clin_scaled = impute_and_scale(clinical, "clinical", self.fold, save_dir=self.staging_dir)
        ##X_meth_scaled = impute_and_scale(X_meth_sel, "methylation", self.fold, save_dir=self.staging_dir)
        X_meth_scaled = impute_and_scale(methyl_df, "methylation", self.fold, save_dir=self.staging_dir)

        # For now, select a random row from methylation data
        ##X_meth_scaled = select_random_methylation_row(X_meth_scaled, selected_features)

        # Ensure both DataFrames have the same index
        X_clin_scaled = X_clin_scaled.reset_index(drop=True)
        X_meth_scaled = X_meth_scaled.reset_index(drop=True)

        if self.model_name == "combined":
            X_input = pd.concat([X_clin_scaled, X_meth_scaled], axis=1)
        elif self.model_name == "clinical":
            X_input = X_clin_scaled
        elif self.model_name == "methylation":
            X_input = X_meth_scaled
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        #st.write('nans: ', X_clin_scaled.isna().sum())
        #st.write("Clinical shape:", X_clin_scaled.shape)
        #st.write("Methylation shape:", X_meth_scaled.shape)

        # Store X_input in the agent for later retrieval (used in SHAP)
        self.X_input = X_input

        # Predict
        predictions = self.model.predict(X_input)
        probs = self.model.predict_proba(X_input)

        # Get class labels
        class_labels = self.model.classes_

        # Create a DataFrame of all class probabilities
        probs_df = pd.DataFrame(probs, columns=class_labels)

        # Add predicted class and confidence score
        probs_df["predicted_class"] = predictions
        probs_df["confidence"] = probs_df.max(axis=1)

        # Reorder columns to put predicted class first
        cols = ["predicted_class", "confidence"] + [col for col in probs_df.columns if col not in ["predicted_class", "confidence"]]
        probs_df = probs_df[cols]

        return probs_df

    def get_X_input(self) -> pd.DataFrame:        

        '''clinical = preprocessing_clinical_data(clinical_df)

        # Load model
        model_path = os.path.join(self.models_dir, f"{self.model_name}_{self.model_type}_fold{self.fold}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = joblib.load(model_path)

        # Include all features expected by model, set as 0 if not present
        expected_features = model.feature_names_in_
        clinical = clinical.reindex(columns=expected_features, fill_value=0)
        # Remove chromosomal features
        clinical = clinical.loc[:, ~clinical.columns.map(is_chromosomal)]

        # Impute and Scale
        X_clin_scaled = impute_and_scale(clinical, "clinical", self.fold, save_dir=self.staging_dir)
        X_meth_scaled = impute_and_scale(methyl_df, "methylation", self.fold, save_dir=self.staging_dir)

        # Ensure both DataFrames have the same index
        X_clin_scaled = X_clin_scaled.reset_index(drop=True)
        X_meth_scaled = X_meth_scaled.reset_index(drop=True)

        if self.model_name == "combined":
            X_input = pd.concat([X_clin_scaled, X_meth_scaled], axis=1)
        elif self.model_name == "clinical":
            X_input = X_clin_scaled
        elif self.model_name == "methylation":
            X_input = X_meth_scaled
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")'''

        if self.X_input.empty:
            raise ValueError("No input data available. Please run the agent first.")

        return self.X_input
    
    def run_shap_analysis(self, clinical, methyl):
        
        # Run SHAP analysis using the trained RF model and input features.
        # Focuses only on clinical + top 2 genomic methylation features.

        # Make sure model was run
        if self.X_input.empty:
            raise ValueError("No input data available. Please run the agent first.")

        # Run SHAP
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_input) # eg. shape: (1, 240, 3)

        return shap_values # Holds 3 arrays, one for each class's explanation

    def get_feature_names(self):
        return self.X_input.columns
    


