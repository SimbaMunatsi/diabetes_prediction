import shap
import numpy as np

def explain_model(model, X_train, X_sample):
    explainer = shap.KernelExplainer(
        model.predict,
        shap.sample(X_train, 100)
    )

    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar"
    )
