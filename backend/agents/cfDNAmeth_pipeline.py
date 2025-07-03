import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import shap
import joblib
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.svm import SVC
from boruta import BorutaPy
from sklearn.decomposition import PCA

# ------------------------------
# CONFIGURATION
# ------------------------------
# base_dir = "/Users/soumitaghosh/work/uhn_nash/graft_cfmedip_cristina/new_analysis_UHN_Hannover_Boruta"
# base_dir = "/home/t119011uhn/code/cfmedipseq_boruta"
current_dir = os.path.dirname(os.path.abspath(__file__))   # current script directory
base_dir = os.path.dirname(current_dir)  

data_dir = os.path.join(base_dir, "data")
results_dir = os.path.join(base_dir, "results")
models_dir = os.path.join(base_dir, "models")
staging_dir = os.path.join(base_dir, "data", "staging")
figures_dir = os.path.join(base_dir, "results", "figures")
features_dir = os.path.join(base_dir, "results", "selected_features")

metrics_log_file = os.path.join(results_dir, "all_metrics_log.csv")
n_splits = 5
random_seed = 42
shap_summary_storage = {}

# ------------------------------
# LOAD AND PROCESS METHYLATION + CLINICAL DATA
# ------------------------------

def compute_fold_changes(df, label_col):
    print("Computing fold changes between disease groups...")
    groups = df[label_col].unique()
    if "Ctrl-LT" not in groups:
        print("Ctrl-LT group not found, skipping fold-change computation.")
        return None

    control = df[df[label_col] == "Ctrl-LT"].drop(columns=[label_col])
    fold_change_df = pd.DataFrame(index=control.columns)

    for group in groups:
        if group == "Ctrl-LT":
            continue
        case = df[df[label_col] == group].drop(columns=[label_col])
        log2_fc = case.mean() - control.mean()
        # fold_change_df[f"log2_FC_{group}_vs_Ctrl-LT"] = log2_fc
        _name = f"log2FC_{group.replace(' ', '_')}_vs_Ctrl_LT"
        fold_change_df[_name] = log2_fc

    fc_path = os.path.join(results_dir, "log2_fold_changes.tsv")
    fold_change_df.to_csv(fc_path, sep="	")
    print(f"Saved fold-change data to {fc_path}")
    return fold_change_df


def preprocessing_methylation_data(save_path=os.path.join(staging_dir, "methylation_processed.npz")):
    print("Loading and preprocessing methylation data...")
    meth = pd.read_csv(
        os.path.join(data_dir, "135_rpkm_corrected_with_header"),
        sep="\t",
        index_col=0
    )
    
    meth['region'] = meth.index.astype(str) + ':' + meth['start'].astype(str) + '-' + meth['stop'].astype(str)
    meth.set_index('region', inplace=True)
    meth.drop(columns=['start', 'stop'], inplace=True)

    all_zeros = (meth == 0).all(axis=1)
    meth = meth[~all_zeros]

    zero_threshold = int(0.7 * meth.shape[1])
    rows_with_70_percent_zeros = (meth == 0).sum(axis=1) >= zero_threshold
    rows_less_than_1 = (meth < 1).all(axis=1)
    meth = meth[~(rows_with_70_percent_zeros | rows_less_than_1)]

    epsilon = 0.001
    meth_log2 = meth.applymap(lambda x: np.log2(x + epsilon))
    meth_log2_T = meth_log2.T

    sample_info = pd.read_csv(os.path.join(data_dir, "135_sample_annotation.txt"), sep="\t", index_col=0)
    final_df = meth_log2_T.join(sample_info, how='inner')

    label_col = "disease"
    y = final_df[label_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_dict = {original: encoded for original, encoded in zip(y, y_encoded)}

    X = final_df.drop(columns=[label_col])

    np.savez_compressed(
        save_path,
        X=X.values,
        y=y_encoded,
        feature_names=np.array(X.columns),
        sample_ids=np.array(X.index)
    )
    print(f"Saved processed methylation data to {save_path}")

    print(f"Methylation shape: {X.shape}")
    compute_fold_changes(pd.concat([X, pd.Series(y, index=final_df.index, name=label_col)], axis=1), label_col)
    return X, y_encoded, le

def load_preprocessed_methylation_data(load_path=os.path.join(staging_dir, "methylation_processed.npz")):
    print(f"📂 Loading preprocessed methylation data from {load_path}...")
    data = np.load(load_path, allow_pickle=True)
    X = pd.DataFrame(data['X'], columns=data['feature_names'], index=data['sample_ids'])
    y = data['y']
    return X, y 

def preprocessing_clinical_data(clinical_df):
    print("Loading and preprocessing clinical data...")
    clinical_df.set_index("new_sampleid_corrected", inplace=True)
    clinical_df_encoded = pd.get_dummies(clinical_df, columns=["sex", "txp.indication"], drop_first=False)
    # Since not all these values may be present after encoding, check first before dropping
    drop_cols = ['cohort', 'previous_id', 'sample', 'disease', 'sex_F']
    cols_to_drop = [col for col in drop_cols if col in clinical_df_encoded.columns]
    clinical_df_encoded.drop(columns=cols_to_drop, inplace=True)

    return clinical_df_encoded

"""print("Loading and preprocessing clinical data...")
    clinical_df = pd.read_csv(os.path.join(data_dir, "135_clinical_info.txt"), sep="\t", encoding='utf-8')
    clinical_df.set_index("new_sampleid_corrected", inplace=True)
    clinical_df_encoded = pd.get_dummies(clinical_df, columns=["sex", "txp.indication"], drop_first=False)
    clinical_df_encoded.drop(columns=['cohort', 'previous_id', 'sample','disease', 'sex_F'], inplace=True)
    return clinical_df_encoded"""

"""print("Loading and preprocessing clinical data...")
    clinical_df.set_index("new_sampleid_corrected", inplace=True)
    clinical_df_encoded = pd.get_dummies(clinical_df, columns=["sex", "txp.indication"], drop_first=False)
    clinical_df_encoded.drop(columns=['cohort', 'previous_id', 'sample','disease', 'sex_F'], inplace=True)
    return clinical_df_encoded"""
    

# ------------------------------
# IMPUTE AND SCALE
# ------------------------------
import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def impute_and_scale(X_test, data_type, fold, save_dir, X_train=None):
    """
    Imputes and scales features. Saves parameters during training, reuses during inference.

    Args:
        X_test (DataFrame): Data to transform (used during inference or test time).
        X_train (DataFrame or None): Training data to fit (None for inference only).
        save_dir (str): Directory to save/load imputer and scaler.
        model_name (str): Unique identifier for the saved parameter files.

    Returns:
        Transformed X_train and X_test as DataFrames (or only X_test if inference mode).
    """
    os.makedirs(save_dir, exist_ok=True)
    imputer_path = os.path.join(save_dir, f"{data_type}_fold{fold}_imputer.pkl")
    scaler_path = os.path.join(save_dir, f"{data_type}_fold{fold}_scaler.pkl")

    if X_train is not None:
        print("Training mode: imputing and scaling, saving parameters...")

        # Fit on training data
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # Save
        joblib.dump(imputer, imputer_path)
        joblib.dump(scaler, scaler_path)

        return (
            pd.DataFrame(X_train_scaled, columns=X_train.columns),
            pd.DataFrame(X_test_scaled, columns=X_test.columns)
        )

    else:
        print("Inference mode: loading saved imputer and scaler...")

        # Load saved parameters
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)

        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)

        return pd.DataFrame(X_test_scaled, columns=X_test.columns)


# ------------------------------
# FEATURE SELECTION (Boruta)
# ------------------------------
def select_features_boruta(X, y, fold, chunk_size=5000, overlap=1000):

    print(f"Starting Boruta feature selection for fold {fold}...")
    selected_features_all = []
    n_features = X.shape[1]
    start_indices = list(range(0, n_features, chunk_size - overlap))

    for i, start in enumerate(start_indices):

        end = min(start + chunk_size, n_features)
        chunk_path = os.path.join(features_dir, f"boruta_selected_chunk{start}_{end}_fold{fold}.txt")
        
        if os.path.exists(chunk_path):
            print(f"Chunk {start} to {end} already processed, skipping...")
            with open(chunk_path, "r") as f:
                selected = [line.strip() for line in f.readlines()]
                selected_features_all.extend(selected)
            continue
        
        chunk = X.iloc[:, start:end]
        print(f"Fold {fold} - Running Boruta on chunk {i+1}/{len(start_indices)}: features {start} to {end}")

        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_seed)
        boruta = BorutaPy(estimator=rf, n_estimators='auto', verbose=0, random_state=random_seed)
        boruta.fit(chunk.values, y)
        selected = chunk.columns[boruta.support_].tolist()
        selected_features_all.extend(selected)

        # Save selected features per chunk
        with open(chunk_path, "w") as f:
            for feat in selected:
                f.write(f"{feat}\n")

    selected_features_all = sorted(set(selected_features_all))

    # Final Boruta on merged selected features
    if selected_features_all:
        print(f"Running final Boruta on {len(selected_features_all)} merged features...")
        X_reduced = X[selected_features_all]
        rf_final = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_seed)
        boruta_final = BorutaPy(estimator=rf_final, n_estimators='auto', verbose=0, random_state=random_seed)
        boruta_final.fit(X_reduced.values, y)
        final_selected = X_reduced.columns[boruta_final.support_].tolist()
        selected_features_all = final_selected

    print(f"Total selected features for fold {fold}: {len(selected_features_all)}")
    return selected_features_all

# ------------------------------
# TRAIN AND EVALUATE
# ------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model_type, fold):
    if model_name == 'rf':
        clf = RandomForestClassifier(random_state=random_seed)
    elif model_name == 'svc':
        clf = SVC(probability=True, kernel='rbf', random_state=random_seed)
    else:
        raise ValueError("Unsupported model name")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr', average='weighted')

    print(f"Model {model_type}-{model_name} trained. AUC: {auc:.4f}")
    try:
        run_shap_analysis(clf, X_train, X_test, model_name, model_type, fold)
    except Exception as e:
        print(f"SHAP failed for {model_type}-{model_name} fold {fold}: {e}")

 
    # Log metrics
    metrics_row = {
        "fold": fold,
        "model_type": model_type,
        "model_name": model_name,
        "auc": auc,
        **{f"{k}_{label}": v for label, metrics in report.items() if isinstance(metrics, dict) for k, v in metrics.items()}
    }
    df_metrics = pd.DataFrame([metrics_row])

    if not os.path.exists(metrics_log_file):
        df_metrics.to_csv(metrics_log_file, index=False)
    else:
        df_metrics.to_csv(metrics_log_file, mode='a', header=False, index=False)

    # Save model
    model_path = os.path.join(models_dir, f"{model_type}_{model_name}_fold{fold}.pkl")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

    return report, auc


# ------------------------------
# SHAP ANALYSIS
# ------------------------------
def run_shap_analysis(model, X_train, X_test, model_name, model_type, fold):
        
    print(fold, X_train.shape, X_test.shape)
    if model_name == 'svc':
        explainer = shap.KernelExplainer(model.predict_proba, X_train)
        raw_shap = explainer.shap_values(X_test)            # list of arrays
        shap_values = np.stack(raw_shap, axis=0)            # (n_classes, samples, features)
        # shap_values = np.transpose(shap_values, (1, 2, 0))  # (samples, features, n_classes)
    else:
        explainer = shap.Explainer(model.predict_proba, X_train)
        raw_shap = explainer(X_test, max_evals=2 * X_test.shape[1] + 1)
        shap_values = raw_shap.values  # already (samples, features, n_classes)

    print(shap_values.shape)  # should be (samples, features, n_classes)

    # print(fold, len(shap_values), len(shap_values[0]), len(shap_values[0][0]))
    
    # shap.summary_plot(shap_values, X_test, show=False)

    n_classes = shap_values.shape[2]
    class_names = [f"Class {i}" for i in range(n_classes)]

    # compute the mean absolute SHAP value per feature, for each class
    # shap_values[:, :, i] → extracts SHAP values for class i, shape: (samples, features)
    # np.abs(...) → takes absolute value of SHAP values (since they can be negative/positive)
    # .mean(axis=0) → averages across samples → result: 1D array of length features
    # {class_names[i]: ...} → stores that 1D array in a dictionary keyed by class name
    # mean_abs_shap -> {"CTRL-LT": array([...264 values...]),
    #                   "TCMR": array([...264 values...]),
    #                   "MASH-LT": array([...264 values...])}
    mean_abs_shap = {
        class_names[i]: np.abs(shap_values[:, :, i]).mean(axis=0)
        for i in range(n_classes)
    }

    # Create DataFrame: rows = features, cols = class-wise mean(|SHAP|)
    shap_df = pd.DataFrame(mean_abs_shap, index=X_test.columns)

    # Add total importance and sort
    shap_df_plot = shap_df.copy()
    shap_df_plot["Total"] = shap_df_plot.sum(axis=1)
    shap_df_plot = shap_df_plot.sort_values("Total", ascending=False).drop(columns="Total")

    # Optional: keep top N features
    top_n = 50
    shap_df_top = shap_df_plot.head(top_n)

    # Plot: Stacked horizontal bar chart
    fig_name = os.path.join(figures_dir, f"shap_{model_type}_{model_name}_fold{fold}.pdf")
    shap_df_top.plot(kind='barh', stacked=True, figsize=(10, 8))
    plt.title("Stacked SHAP Feature Importance Across Classes")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.close()
    # plt.show()

    # shap_summary_storage = {
    #       "combined_rf": [
    #           shap_df,  # features x class-wise mean(|SHAP|) for fold 1
    #           shap_df,  # features x class-wise mean(|SHAP|) for fold 1
    #           ...
    #       ]
    # }
    model_key = f"{model_type}_{model_name}"
    # shap_summary_storage.setdefault(model_key, []).append(np.abs(shap_values.values).mean(axis=0))
    shap_summary_storage.setdefault(model_key, []).append(shap_df)

# ------------------------------
# PLOT PCA
# ------------------------------

def plot_pca(X_input, labels, title, filename):  # updated with variance info
    X = X_input.copy()
    X = X.fillna(0)
    print(f"Plotting PCA: {title}")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_ * 100
    pc1_label = f"PC1 ({var_ratio[0]:.1f}% var)"
    pc2_label = f"PC2 ({var_ratio[1]:.1f}% var)"
    df_pca = pd.DataFrame(X_pca, columns=[pc1_label, pc2_label])
    df_pca['label'] = labels
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        subset = df_pca[df_pca['label'] == label]
        plt.scatter(subset[pc1_label], subset[pc2_label], label=label, alpha=0.6)
    plt.title(title)
    plt.xlabel(pc1_label)
    plt.ylabel(pc2_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename))
    plt.close()


# ------------------------------
# SHAP MEAN SUMMARY
# ------------------------------
def _save_mean_shap_summary():
    print("Saving mean SHAP summaries across folds...")
    for model_key, shap_list in shap_summary_storage.items():
        mean_shap = np.mean(np.vstack(shap_list), axis=0)
        plt.figure()
        plt.bar(range(len(mean_shap)), mean_shap)
        plt.title(f"Mean SHAP Summary: {model_key}")
        plt.savefig(os.path.join(figures_dir, f"mean_shap_{model_key}.pdf"))
        plt.close()


def save_mean_shap_summary(n=100):

    print("Saving mean SHAP summaries across folds...")
    
    for model_key in shap_summary_storage:
        all_dfs = shap_summary_storage[model_key]  # list of DataFrames
        mean_shap_df = sum(all_dfs) / len(all_dfs)  # element-wise average
        mean_shap_df["mean_importance"] = mean_shap_df.mean(axis=1)
        mean_shap_df = mean_shap_df.sort_values("mean_importance", ascending=False)

        # Sort and select top n features
        top_features = mean_shap_df.sort_values("mean_importance", ascending=False).head(n)

        # Create the plot
        plt.figure(figsize=(8, 10))
        plt.barh(top_features.index, top_features["mean_importance"])
        plt.xlabel("Mean |SHAP| Value")
        plt.title(f"Top {n} Features by Mean Absolute SHAP Importance")
        plt.gca().invert_yaxis()  # Highest at the top

        # Save as PDF
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"mean_shap_{model_key}.pdf"), dpi=300)
        plt.close()

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def cross_validate_models(clinical, methylation, labels):

    print("Starting cross-validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    """
    skf_path = os.path.join(results_dir, "stratified_kfold.pkl")
    joblib.dump(skf, skf_path)
    print(f"StratifiedKFold object saved to {skf_path}")
    """

    def run_fold(fold, train_idx, test_idx):
        print(f"\n----- Fold {fold+1}/{n_splits} -----")

        y_train, y_test = labels[train_idx], labels[test_idx]
        X_train_clin, X_test_clin = clinical.iloc[train_idx], clinical.iloc[test_idx]
        X_train_meth, X_test_meth = methylation.iloc[train_idx], methylation.iloc[test_idx]

        # PCA for full input (before selection)
        plot_pca(X_train_clin, y_train, f"Clinical PCA - Fold {fold}", f"pca_clinical_fold{fold}.pdf")
        # plot_pca(X_train_meth, y_train, f"Methylation PCA - Fold {fold}", f"pca_methylation_fold{fold}.pdf")
        print(f"\n----- Fold {fold+1}/{n_splits} -----")

        # Boruta
        selected_features_path = os.path.join(features_dir, f"boruta_selected_features_fold{fold}.txt")
        if os.path.exists(selected_features_path):
            with open(selected_features_path, "r") as f:
                selected_features = [line.strip() for line in f.readlines()]
            print(f"Selected features already exist for fold {fold}, skipping Boruta...")
        else:
            print(f"Running Boruta for fold {fold}...")
            # Select features using Boruta
            selected_features = select_features_boruta(X_train_meth, y_train, fold)
        
        # Save selected features
        with open(selected_features_path, "w") as f:
            for feat in selected_features:
                f.write(f"{feat}\n")
        print(f"Selected features saved to {selected_features_path}")

        # Filter methylation data based on selected features
        X_train_meth_sel, X_test_meth_sel = X_train_meth[selected_features], X_test_meth[selected_features]

        X_train_clin_scaled, X_test_clin_scaled = impute_and_scale(X_test_clin, "clinical", fold, save_dir=staging_dir, X_train=X_train_clin)
        X_train_meth_scaled, X_test_meth_scaled = impute_and_scale(X_test_meth_sel, "methylation", fold, save_dir=staging_dir, X_train=X_train_meth_sel)

        X_train_combined = pd.concat([X_train_clin_scaled, X_train_meth_scaled], axis=1)
        plot_pca(X_train_meth_scaled[selected_features], y_train, f"Boruta-selected Methylation PCA - Fold {fold}", f"pca_methylation_boruta_fold{fold}.pdf")
        plot_pca(X_train_combined, y_train, f"Combined PCA - Fold {fold}", f"pca_combined_fold{fold}.pdf")
        X_test_combined = pd.concat([X_test_clin_scaled, X_test_meth_scaled], axis=1)

        for model_type, Xtr, Xte in zip([
            "clinical", "methylation", "combined"],
            [X_train_clin_scaled, X_train_meth_scaled, X_train_combined],
            [X_test_clin_scaled, X_test_meth_scaled, X_test_combined]):

            for model_name in ["rf", "svc"]:
                print(f"Training {model_type} model with {model_name.upper()}")
                report, auc = train_and_evaluate(Xtr, Xte, y_train, y_test, model_name, model_type, fold)
                joblib.dump({"report": report, "auc": auc}, os.path.join(results_dir, f"metrics_{model_type}_{model_name}_fold{fold}.pkl"))

    expected_models = [
        "clinical_rf", "clinical_svc",
        "methylation_rf", "methylation_svc",
        "combined_rf", "combined_svc"
    ]

    Parallel(n_jobs=5)(
        delayed(run_fold)(fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(skf.split(clinical, labels))
    )

    save_mean_shap_summary()

# ------------------------------
# RUN SCRIPT
# ------------------------------
if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    print("Starting pipeline...")
    methylation_path = os.path.join(staging_dir, "methylation_processed.npz")

    if os.path.exists(methylation_path):
        methylation, labels = load_preprocessed_methylation_data(methylation_path)
    else:
        methylation, labels, _ = preprocessing_methylation_data(methylation_path)

    clinical = preprocessing_clinical_data()
    cross_validate_models(clinical, methylation, labels)
