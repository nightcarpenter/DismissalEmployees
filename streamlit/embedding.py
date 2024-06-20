import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.models import Model

'''
    Нуден для того что бы можно было кодировать категориальные признаки
    по технологии embedding, с которой хорошо работают модели градиентного бустинга
'''
class EmbeddingEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_dim=2, handle_unknown='ignore'):
        self.embedding_dim = embedding_dim
        self.handle_unknown = handle_unknown
        # Словарь уникальных значений категорий
        self.embeddings_ = {}
        # Словарь моделей встраиваний
        self.models_ = {}
        # Подавляем вывод журнала
        tf.get_logger().setLevel('ERROR')

    def fit(self, X, y=None):
        # Обучение моделей встраиваний для каждого столбца
        original_log_level = tf.get_logger().level

        # Проходим по каждому столбцу данных
        for i, col in enumerate(X.T):
            # Находим и сохраняем уникальные значения в столбце
            unique_vals = np.unique(col)
            self.embeddings_[i] = unique_vals
            # Входной слой для модели
            input_layer = Input(shape=(1,))
            # Слой встраивания
            embedding_layer = Embedding(input_dim=len(unique_vals) + 1,
                                        output_dim=self.embedding_dim,
                                        input_length=1)(input_layer)
            # Развертывание вектора встраивания в плоский массив
            flatten_layer = Flatten()(embedding_layer)
            # Создаем, компилируем и сохраняем модель для текущего столбца
            model = Model(inputs=input_layer, outputs=flatten_layer)
            model.compile(optimizer='rmsprop', loss='mse')
            self.models_[i] = model
        # Восстанавливаем оригинальный уровень логирования TensorFlow
        tf.get_logger().setLevel(original_log_level)
        return self

    def transform(self, X):
        # Преобразование данных с использованием обученных моделей встраиваний.
        original_log_level = tf.get_logger().level

        # Список для хранения преобразованных данных
        X_transformed = []
        # Проходим по каждому столбцу данных
        for i, col in enumerate(X.T):
            # Получаем модель для текущего столбца
            model = self.models_[i]

            # Список для хранения индексов категорий
            col_indices = []
            # Преобразуем каждое значение в столбце в индекс
            for val in col:
                if val in self.embeddings_[i]:
                    col_indices.append(np.where(self.embeddings_[i] == val)[0][0])
                else:
                    # Не забываем про индекс для неизвестной категории
                    if self.handle_unknown == 'ignore':
                        col_indices.append(len(self.embeddings_[i]))
                    else:
                        raise ValueError(f"Unknown category '{val}' encountered in column {i}.")

            # Преобразуем список индексов в массив
            col_indices = np.array(col_indices)
            # Получаем встраивания для индексов
            col_embedded = model.predict(col_indices, verbose=0)
            # Добавляем встраивания в список
            X_transformed.append(col_embedded)

        # Возвращаем преобразованные данные как один массив
        return np.hstack(X_transformed)

    def get_feature_names_out(self, input_features=None):
        # Формирование списка имен новых признаков после встраивания
        feature_names = []
        for i in range(len(self.embeddings_)):
            for dim in range(self.embedding_dim):
                feature_names.append(f"col_{i}_dim_{dim}")
        return np.array(feature_names)


if __name__ == "__main__":
    emb = EmbeddingEncoder()
