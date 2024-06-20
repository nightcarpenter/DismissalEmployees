import streamlit as st

from page_form import InputFormPage
from models import Models
from shap_show import ShapShow
from calculations import Calculations


class App:
    def __init__(self):
        st.set_page_config(layout="wide")
        self.models = Models()
        self.shap_show_classifier = ShapShow(self.models.pipe_classifier,
                                             self.models.X_quit,
                                             self.models.y_quit)
        self.calculations = Calculations(self.models, self.shap_show_classifier)
        self.pages = {
            "Форма ввода": InputFormPage(self.calculations)
        }


    def run(self):
        st.sidebar.title("Навигация")
        page = st.sidebar.radio("Выберите страницу", list(self.pages.keys()))

        # Отображаем выбранную страницу
        self.pages[page].render()


if __name__ == "__main__":
    app = App()
    app.run()