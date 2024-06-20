import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import shap
from sklearn.preprocessing import (OneHotEncoder)
from io import BytesIO


# Класс для анализа важности признаков с помощью shap
class ShapShow:
    def __init__(self, pipeline, X_train, y_train):
        self.pipeline = pipeline
        if 'dept' in X_train.columns:
            X_train = X_train.drop(columns='dept')
        self.X_train = X_train
        self.y_train = y_train

        self.feature_names = self.X_train.columns.tolist()
        self.cat_feature_names = self.X_train.select_dtypes(include=['object']).columns.tolist()
        self.num_feature_names = self.X_train.select_dtypes(exclude=['object']).columns.tolist()

        self.create_shap_values()


    # Метод для создания shap-значений
    def create_shap_values(self):
        # Подготавливаем данные
        self.X_train_preprocessed = self.pipeline.named_steps['preprocessor'].transform(self.X_train)

        # Сохраняем в переменную название использовавшегося кодировщика
        cat_encoding = self.pipeline['preprocessor'].transformers_[0][1].steps[1][1]

        # Если данные были закодированы OneHotEncoder
        if isinstance(cat_encoding, OneHotEncoder):
            cat_cols = self.pipeline['preprocessor'].transformers_[0][1][1].get_feature_names_out().tolist()
            num_cols = self.pipeline['preprocessor'].transformers_[1][2]
            cols = cat_cols + num_cols
            self.X_train_preprocessed = pd.DataFrame(self.X_train_preprocessed, columns=cols)
            self.X_train_preprocessed.columns = self.X_train_preprocessed.columns.str.replace('x0', 'level')
            self.X_train_preprocessed.columns = self.X_train_preprocessed.columns.str.replace('x1', 'workload')
            self.X_train_preprocessed.columns = self.X_train_preprocessed.columns.str.replace('x2', 'last_year_promo')
            self.X_train_preprocessed.columns = self.X_train_preprocessed.columns.str.replace('x3', 'last_year_violations')

        # Если данные были закодированы другими колировщиками
        else:
            # Получаем названия закодированных колонок в нужном порядке
            cat_cols = self.pipeline['preprocessor'].transformers_[0][2]
            num_cols = self.pipeline['preprocessor'].transformers_[1][2]
            cols = cat_cols + num_cols
            # Создаём датафрейм с закодированными признаками
            self.X_train_preprocessed = pd.DataFrame(self.X_train_preprocessed, columns=cols)

        # Создание explainer на основе модели
        self.explainer = shap.TreeExplainer(self.pipeline.named_steps['model'])
        # Вычисление значений SHAP для предварительно обработанного тестового набора данных
        self.shap_values = self.explainer.shap_values(self.X_train_preprocessed)

        # Создаём новый shap_values и shap_explanation, для кодировщиков,
        # которые увеличивают количество категориальных признаков
        if (isinstance(cat_encoding, OneHotEncoder)):
            # Получаем русские названия колонок
            feature_mapping = self.get_feature_names_mapping()
            feature_names_rus = [feature_mapping.get(col, col) for col in self.X_train.columns.tolist()]

            # Восстанавливаем исходное количество категориальных признаков
            self.shap_values = self.aggregate_shap_values(self.shap_values,
                                                          self.X_train_preprocessed.columns.tolist(),
                                                          self.feature_names,
                                                          self.num_feature_names)

            # Создание объекта SHAP Explanation, с объединёнными значениями shap_values
            self.shap_exp = shap.Explanation(values=self.shap_values,
                                             base_values=self.explainer.expected_value,
                                             feature_names=feature_names_rus,
                                             data=self.X_train.values)

        else:
            # Создание объекта SHAP Explanation, с оригинальными значениями shap_values
            self.shap_exp = shap.Explanation(values=self.shap_values,
                                             base_values=self.explainer.expected_value,
                                             feature_names=self.X_train_preprocessed.columns.tolist(),
                                             data=self.X_train_preprocessed.values)

    # Метод для объединения значений SHAP для категориальных признаков
    def aggregate_shap_values(self, shap_values, encoded_feature_names, original_feature_names, numerical_features):
        # Создаём двумерный массив нулей, размер: количество строк, реальное количество категориальных признаков
        aggregated_shap_values = np.zeros((shap_values.shape[0], len(original_feature_names)))

        # Проходим циклом по всем признакам
        for i, original_feature in enumerate(original_feature_names):
            # Для количественных признаков просто копируем значения SHAP
            if original_feature in numerical_features:
                mask = np.array(encoded_feature_names) == original_feature
                aggregated_shap_values[:, i] = shap_values[:, mask].sum(axis=1)
            # Для категориальных признаков суммируем значения SHAP по закодированным признакам
            else:
                mask = np.array(
                    [encoded_feature.startswith(original_feature) for encoded_feature in encoded_feature_names])
                # Проверка, есть ли соответствие
                if mask.sum() > 0:
                    aggregated_shap_values[:, i] = shap_values[:, mask].sum(axis=1)
                else:
                    raise ValueError(f"No encoded features found for original feature '{original_feature}'")

        return aggregated_shap_values

    def show_waterfall(self, index):
        try:
            # Проверка на наличие данных
            if index >= len(self.shap_exp):
                print("Индекс выходит за пределы массива данных.")
                return None

            # Получаем начальные данные для графика
            original_shap_values = self.shap_exp[index].values

            # print("Original SHAP values shape:", original_shap_values.shape)
            # print("original_shap_values.sum():", original_shap_values.sum())
            # print("self.y_train.mean().values[0]:", self.y_train.mean().values[0])

            base_rate = self.y_train.mean().values[0]
            predicted_prob = self.pipeline.predict_proba(self.X_train.loc[[index]])[:, 1][0]

            # Вычисляем коэффициент масштабирования
            total_effect = original_shap_values.sum()
            scaling_factor = (predicted_prob - base_rate) / total_effect

            # Преобразуем значения
            converted_shap_values = original_shap_values * scaling_factor

            # Создаём массив для графика водопада
            waterfall_values = np.concatenate([[base_rate], converted_shap_values, [predicted_prob]])

            # Создаём объект SHAP Explanation для графика
            new_shap_exp = shap.Explanation(
                values=converted_shap_values,
                base_values=base_rate,
                feature_names=self.shap_exp.feature_names,
                data=self.shap_exp.data[index]
            )

            # Строим график водопада
            shap.waterfall_plot(new_shap_exp, show=False)
            plt.gcf().set_size_inches(12, 8)
            plt.gca().tick_params(labelsize=10)
            plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
            for text in plt.gcf().findobj(match=plt.Text):
                text.set_fontsize(12)

            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()
            return buf

        except Exception as e:
            print(f"Ошибка show_waterfall: {e}")
            return None

    def update_with_new_data(self, new_X_value, new_y_value):
        try:
            # Объединяем старые данные с новыми
            self.X_train = pd.concat([self.X_train, new_X_value], ignore_index=True)

            self.create_shap_values()

            # Возвращаем индекс новой записи
            new_index = len(self.X_train) - 1

            return new_index

        except Exception as e:
            print(f"Ошибка update_with_new_data: {e}")
            return None

    def get_feature_names_mapping(self):
        return {
            'dept': 'отдел',
            'level': 'должность',
            'workload': 'загрузка',
            'employment_years': 'срок работы',
            'last_year_promo': 'повышение',
            'last_year_violations': 'нарушения',
            'supervisor_evaluation': 'оценка',
            'salary': 'зарплата',
            'satisfaction': 'счастье'
        }

if __name__ == "__main__":
    ss = ShapShow()