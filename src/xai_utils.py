"""XAI helper functions for SHAP, LIME, and PDP visualizations."""

import numpy as np
import matplotlib.pyplot as plt


def shap_summary(shap_values, X, max_display=15, title='SHAP Summary'):
    import shap
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    return fig


def shap_waterfall(shap_values, idx=0, title='SHAP Waterfall — Single Prediction'):
    import shap
    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[idx], show=False)
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    return fig


def lime_explain(model, X_train, X_instance, feature_names, mode='regression'):
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, mode=mode, random_state=42
    )
    exp = explainer.explain_instance(X_instance, model.predict, num_features=10)
    return exp


def pdp_plot(model, X, features, feature_names=None, grid_resolution=50):
    from sklearn.inspection import PartialDependenceDisplay
    fig, ax = plt.subplots(figsize=(14, 5))
    PartialDependenceDisplay.from_estimator(
        model, X, features=features,
        feature_names=feature_names,
        grid_resolution=grid_resolution,
        ax=ax, kind='both'
    )
    plt.tight_layout()
    return fig
