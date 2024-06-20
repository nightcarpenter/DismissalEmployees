# Основные библиотеки
import numpy as np
import pandas as pd

# Импорт модулей для сэмплирования и для создания пайплайна
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# Импорт модулей для кодирования и масштабирования
from sklearn.preprocessing import (LabelEncoder,
                                   OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler,
                                   MinMaxScaler,
                                   RobustScaler,
                                   FunctionTransformer)

# Импорт модулей для создания пайплайна
from sklearn.compose import (ColumnTransformer, make_column_selector)

# Импорт модуля для работы с пропусками
from sklearn.impute import SimpleImputer, KNNImputer

# Импорт модулей из catboost
from catboost import CatBoostClassifier, CatBoostRegressor


# Импорт из файлов
from constants import *
from binary_encoder import TargetEncoderClass

class Models:
    def __init__(self):
        self.target_encoder_class = TargetEncoderClass()

        # Загружаем данные и создаём модели
        self.load_datasets()
        self.pipe_classifier = self.create_classifier_model()
        self.pipe_regressor = self.create_regressor_model()

        # Обучаем модель предсказывать удовлетворённость работой
        self.pipe_regressor.fit(self.X_satisfaction, self.y_satisfaction)

        # Добавляем предсказанный признак в датафрейм для предсказания увольнения
        self.X_quit['satisfaction'] = self.pipe_regressor.predict(self.X_quit)

        # Убираем "dept" из датафрейма, он не нужен для предсказания увольнения
        if 'dept' in self.X_quit.columns:
            self.X_quit = self.X_quit.drop(columns='dept')

        # Обучаем модель предсказывать увольнение
        self.pipe_classifier.fit(self.X_quit, self.y_quit)


    def load_datasets(self):
        self.satisfaction = pd.read_csv('datasets/train_job_satisfaction_rate.csv')
        self.quit = pd.read_csv('datasets/train_quit.csv')

        # Удаляем id, удаляем дубликаты, разбиваем на тренировочную и тестовую выборки
        self.satisfaction = self.satisfaction.drop(columns='id')
        self.satisfaction = self.satisfaction.drop_duplicates().reset_index(drop=True)
        self.X_satisfaction = self.satisfaction.drop(columns=['job_satisfaction_rate'])
        self.y_satisfaction = self.satisfaction['job_satisfaction_rate'].to_frame()

        # Удаляем id, удаляем дубликаты, разбиваем на тренировочную и тестовую выборки, кодируем target
        self.quit = self.quit.drop(columns='id')
        self.quit = self.quit.drop_duplicates().reset_index(drop=True)
        self.X_quit = self.quit.drop(columns=['quit'])
        self.y_quit = self.quit['quit'].to_frame()
        self.y_quit = self.target_encoder_class.encoding_target_train(self.y_quit)

        # print('Размеры выборок satisfaction:')
        # print(self.X_satisfaction.shape)
        # print(self.y_satisfaction.shape)
        # print('Размеры выборок quit:')
        # print(self.X_quit.shape)
        # print(self.y_quit.shape)


    def create_classifier_model(self):
        # Создаём пайплайн по заданным параметрам
        cat_pipe = ImbPipeline(
            [
                ('simple_imputer_before_encoder', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')),
                ('simple_imputer_after_encoder', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
            ]
        )

        num_pipe = ImbPipeline(
            [
                ('simple_imputer_before_scaler', SimpleImputer(missing_values=np.nan, strategy='median')),
                ('scaler', StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ('cat', cat_pipe, make_column_selector(dtype_include=object)),
                ('num', num_pipe, make_column_selector(dtype_include=np.number))
            ],
            remainder='passthrough'
        )

        best_pipe_classifier = ImbPipeline(
            [
                ('preprocessor', preprocessor),
                ('sampler', RandomUnderSampler(random_state=RANDOM_STATE)),
                ('model', CatBoostClassifier(random_seed=RANDOM_STATE,
                                             iterations=198,
                                             verbose=False,
                                             bagging_temperature=0.489742,
                                             depth=2,
                                             l2_leaf_reg=0.746038,
                                             learning_rate=0.094673,
                                             random_strength=0.677138))
            ]
        )

        return best_pipe_classifier


    def create_regressor_model(self):
        # Создаём пайплайн по заданным параметрам
        cat_encoding = ImbPipeline(
            [
                ('simple_imputer_before_cat', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                ('cat_encoder', OneHotEncoder(drop='first', handle_unknown='ignore')),
                ('simple_imputer_after_cat', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
            ]
        )

        num_scalling = ImbPipeline(
            [
                ('simple_imputer_before_num', SimpleImputer(missing_values=np.nan, strategy='median')),
                ('num_encoder', MinMaxScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ('cat', cat_encoding, make_column_selector(dtype_include=object)),
                ('num', num_scalling, make_column_selector(dtype_include=np.number))
            ],
            remainder='passthrough'
        )

        best_pipe_regressor = ImbPipeline(
            [
                ('preprocessor', preprocessor),
                ('model', CatBoostRegressor(random_state=RANDOM_STATE,
                                            iterations=1139,
                                            verbose=False,
                                            bagging_temperature=0.264570,
                                            depth=7,
                                            l2_leaf_reg=0.245743,
                                            learning_rate=0.030131,
                                            random_strength=0.323693))
            ]
        )

        return best_pipe_regressor


