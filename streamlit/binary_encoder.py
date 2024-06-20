import numpy as np
from sklearn.preprocessing import LabelEncoder


from constants import *


class TargetEncoderClass:
    def __init__(self):
        self.le = LabelEncoder()

    def encoding_target_train(self, y_train):
        y_train = y_train.copy()

        if set(y_train['quit'].unique()) == set(['no', 'yes']):
            # Выводим первые строки датафрейма до кодирования целевого признака, для контроля
            # print('\ny_train_quit до кодировапния целевого признака:')
            # print(y_train.sample(n=3, random_state=RANDOM_STATE + 2))
            # Присваиваем целевому признаку customer_activity значения 0 и 1
            self.le.fit(y_train['quit'])
            self.le.classes_ = np.array(['no', 'yes'])
            y_train['quit'] = self.le.transform(y_train['quit'])
            # print('\ny_train_quit после кодирования целевого признака:')
            # print(y_train.sample(n=3, random_state=RANDOM_STATE + 2))
        elif set(y_train['quit'].unique()) == set([0, 1]):
            print('\ny_train_quit уже закодирован:')
            print(y_train.sample(n=3, random_state=RANDOM_STATE))
        else:
            print('\nЗначения целевого признака в y_train несоответствуют ожидаемым:')
            print(y_train.sample(n=3, random_state=RANDOM_STATE))

        # Возвращаем датафрейм с закодированным целевым признаком
        return y_train

    def encoding_target_test(self, y_test):
        y_test = y_test.copy()

        if set(y_test['quit'].unique()) == set(['no', 'yes']):
            # Выводим первые строки датафрейма до кодирования целевого признака, для контроля
            # print('\ny_test_quit до кодировапния целевого признака:')
            # print(y_test.sample(n=3, random_state=RANDOM_STATE))

            try:
                y_test['quit'] = self.le.transform(y_test['quit'])
                # print('\ny_test_quit после кодирования целевого признака:')
                # print(y_test.sample(n=3, random_state=RANDOM_STATE))
            except:
                print('\nLabelEncoder ещё не обучен! Вначале закодируйте y_train_quit.')

        elif set(y_test['quit'].unique()) == set([0, 1]):
            print('\ny_test_quit уже закодирован:')
            print(y_test.sample(n=3, random_state=RANDOM_STATE))
        else:
            print('\nЗначения целевого признака в y_test несоответствуют ожидаемым:')
            print(y_test.sample(n=3, random_state=RANDOM_STATE))

        # Возвращаем датафрейм с закодированным целевым признаком
        return y_test


if __name__ == "__main__":
    te = TargetEncoderClass()