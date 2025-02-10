## Разработка информационной системы, предсказывающей увольнение сотрудников

### [Презентация проекта] (https://github.com/nightcarpenter/DismissalEmployees/blob/main/preza_itmo_first_wave_jmlc.pdf)

### Краткое описание проекта

Заказчик предоставил данные с характеристиками сотрудников. Среди них есть уровень удовлетворённости сотрудника работой в компании. Эту информацию получили из форм обратной связи: сотрудники заполняют тест-опросник, и по его результатам рассчитывается доля их удовлетворённости от 0 до 1, где 0 – совершенно не удовлетворён, 1 – полностью удовлетворён.

Собирать данные такими опросниками не легко, поскольку компания достаточно большая, но удовлетворённость работой может влиять на отток сотрудников, а предсказание оттока — одна из важнейших задач HR-аналитиков. Главной целью проекта является разработка модели, максимально точно предсказывающей вероятность увольнения сотрудника из компании, на основании тех данных, которые были предоставлены заказчиком.

Одно из требований заказчика – разработка и тестирование нескольких предсказывающих моделей, с последующим выбором наилучшей из них.

### Задачи проекта

1. Разработка модели, предсказывающей уровень удовлетворённости сотрудников работой, на основе данных, предоставленных заказчиком. Модель оценивается метрикой SMAPE (Symmetric Mean Absolute Percentage Error). Критерий успеха: SMAPE ≤ 15.
2. Составление портрета «уволившегося сотрудника».
3. Проверка гипотезы о том, что уровень удовлетворённости сотрудника работой влияет на вероятность его увольнения.
4. Разработка модели, предсказывающей увольнение сотрудников, на основе данных заказчика, и предсказанного уровня удовлетворённости работой. Модель оценивается метрикой ROC-AUC. Критерий успеха: ROC-AUC ≥ 91.
5. Разработка интерфейса для демонстрации работы модели.

### Этапы работы

#### Часть 1: Предсказание удовлетворённости сотрудников работой.
1. Загрузка данных, изучение общей информации.
2. Предобработка данных.
3. Исследовательский анализ.
4. Корреляционный анализ.
5. Создание пайплайна. 
6. Тестирование нескольких моделей МО, поиск оптимальных параметров 
для каждой модели. Оценка результатов и выбор лучшей модели.

#### Часть 2: Предсказание увольнения сотрудников из компании.
1. Загрузка данных, изучение общей информации.
2. Предобработка данных.
3. Исследовательский анализ.
4. Составление портрета уволившегося сотрудника.
5. Проверка гипотезы о том, что уровень удовлетворённости сотрудника 
работой влияет на вероятность его увольнения.
6. Корреляционный анализ.
7. Добавление предсказанного признака удовлетворённости работой в 
датафрейм для предсказания увольнения сотрудника.
8. Создание пайплайна. 
9. Тестирование нескольких моделей МО, поиск оптимальных параметров 
для каждой модели. Оценка результатов и выбор лучшей модели.

#### Часть 3: Анализ модели, предсказывающей увольнение.
1. Оценка влияния входных признаков на предсказания модели.
2. Анализ зависимости предсказаний модели от выбранного порогового 
значения.
3
3. Оценка влияния значений входных признаков на конкретные 
предсказания.

#### Часть 4: Создание интерфейса для демонстрации работы модели для 
заказчика.

### Данные для исследования
- id – уникальный идентификатор сотрудника
- dept – отдел, в котором работает сотрудник
- level – уровень занимаемой должности
- workload – уровень загруженности сотрудника
- employment_years – длительность работы в компании (в годах)
- last_year_promo – показывает, было ли повышение за последний год
- last_year_violations – показывает, нарушал ли сотрудник трудовой 
договор за последний год
- supervisor_evaluation – оценка качества работы сотрудника, которую дал 
руководитель
- salary – ежемесячная зарплата сотрудника
- job_satisfaction_rate – уровень удовлетворённости сотрудника работой в 
компании, целевой признак
- quit – увольнение сотрудника, целевой признак
