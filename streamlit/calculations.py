import pandas as pd
import numpy as np
import joblib


class Calculations:
    def __init__(self, models, shap_show_classifier):
        self.models = models
        self.shap_show_classifier = shap_show_classifier


    def predict_and_create_plot(self, new_data):
        try:
            quit_proba = self.models.pipe_classifier.predict_proba(new_data)[:, 1][0]

            new_X_value = new_data.copy()
            new_y_value = pd.DataFrame({"quit": [quit_proba]})
            if 'dept' in new_X_value.columns:
                new_X_value = new_X_value.drop(columns='dept')
            new_index = self.shap_show_classifier.update_with_new_data(new_X_value, new_y_value)

            buf = self.shap_show_classifier.show_waterfall(new_index)

            return quit_proba, buf

        except Exception as e:
            print(f"Ошибка в predict_and_create_plot: {e}")
            return None, None


if __name__ == "__main__":
    calculations = Calculations()