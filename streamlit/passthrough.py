from sklearn.base import BaseEstimator, TransformerMixin

'''
    Создаём класс-пустышку для семплера.
    Нужен для того, что бы сэмплер не выдавал ошибке при попытке вызвать fit и predict,
    при значении paththrough.
'''
class Passthrough(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X