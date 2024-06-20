import streamlit as st
import pandas as pd

class InputFormPage:
    def __init__(self, calculations):
        self.calculations = calculations

    def render(self):
        # Добавляем CSS стили
        st.markdown(
            """
            <style>
            select, input {
                padding: 0px !important;
            }
            .stSlider > {
                padding: 0px 0px;
            }
            .stButton button {
                font-size: 14px;
            }
            p.answer {
                font-size: 28px;
                font-weight: bold;
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<h2 style='text-align: center;'>Расчёт вероятности увольнения сотрудника</h2>",
                    unsafe_allow_html=True)

        # Разделяем страницу на две колонки
        col1, col2 = st.columns([1, 3])

        with col1:
            with st.form("employee_data_form"):
                dept = st.selectbox("Отдел", ["sales", "technology", "purchasing", "marketing", "hr"], key="dept")
                level = st.selectbox("Уровень занимаемой должности", ["junior", "middle", "sinior"], key="level")
                workload = st.selectbox("Уровень загрузки работой", ["low", "medium", "high"], key="workload")
                last_year_violations = st.selectbox("Нарушения трудового договора за последний год", ["no", "yes"],
                                                    key="last_year_violations")
                last_year_promo = st.selectbox("Повышение за последний год", ["no", "yes"], key="last_year_promo")
                salary = st.slider("Зарплата", min_value=10000, max_value=100000, step=100, key="salary")
                employment_years = st.slider("Срок работы в компании", min_value=1, max_value=10,
                                             key="employment_years")
                supervisor_evaluation = st.slider("Оценка работы сотрудника", min_value=1, max_value=5,
                                                  key="supervisor_evaluation")
                satisfaction = st.slider("Удовлетворённость работой", min_value=0.0, max_value=1.0, key="satisfaction")

                # Кнопка для отправки формы
                submit_button = st.form_submit_button(label="Рассчитать вероятность")

                # Обработка данных после нажатия кнопки
                if submit_button:
                    # Преобразование данных формы в DataFrame
                    data = {
                        'dept': [dept],
                        'level': [level],
                        'workload': [workload],
                        'employment_years': [employment_years],
                        'last_year_promo': [last_year_promo],
                        'last_year_violations': [last_year_violations],
                        'supervisor_evaluation': [supervisor_evaluation],
                        'salary': [salary],
                        'satisfaction': [satisfaction]
                    }
                    df = pd.DataFrame(data)

                    # Вызов метода для предсказания и получения графика
                    quit_pred_proba, buf = self.calculations.predict_and_create_plot(df)

                    # Отображение графика
                    with col2:
                        # Пример графика
                        if buf:
                            # Отображение графика
                            st.image(buf, use_column_width=True)
                        else:
                            st.write("Ошибка при построении графика.")

                        st.markdown(f"<p class='answer'>Вероятность увольнения: {quit_pred_proba:.2f}</p>",
                                    unsafe_allow_html=True)