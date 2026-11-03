# Модуль 1 — Titanic: EDA и бинарная классификация

**Автор:** Власов Дмитрий Сергеевич, АСОиУб-23-2  
**Преподаватель:** Спирин Игорь Сергеевич  
**Дата обучения:** 02.10.2026

## Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Время fit, с |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.7833 | 0.6812 | 0.7287 | 0.8489 | 0.0148 |
| Decision Tree | 0.7821 | 0.7419 | 0.6667 | 0.7023 | 0.8132 | 0.0065 |
| Random Forest | 0.8045 | 0.8400 | 0.6087 | 0.7059 | 0.8482 | 0.2761 |


Стратифицированный split 80:20, random_state=42: 712 строк для обучения и
179 для оценки. Все модели обучены на одной части. Лучший ROC-AUC на этом
разбиении: **Logistic Regression, 0.8489**. Отдельный test.csv Kaggle
содержит 418 неразмеченных строк и используется только для submission.csv.
Время fit зависит от машины; значения таблицы относятся к выполненному запуску.

## Просмотр и повторный запуск

Откройте `report.html` для просмотра готового отчёта без Python. Чтобы повторить
расчёт, используйте Python 3.12 и отдельное окружение:

```bash
cd module-1-titanic
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace notebook.ipynb
```

## Быстрый локальный инференс

```python
from titanic_utils import load_local_artifacts, predict_passenger

bundle = load_local_artifacts("models")
passenger = dict(Pclass=1, Sex="female", Age=25, SibSp=1, Parch=0, Fare=100, Embarked="C")
prediction, probability = predict_passenger(
    passenger, bundle["lr_model"], bundle["scaler"], bundle["feature_cols"],
    preprocessor=bundle["preprocessor"],
)
print(prediction, probability)
```
## Структура

```text
module-1-titanic/
├── notebook.ipynb           # 14 разделов с выполненными ячейками
├── README.md
├── requirements.txt
├── data/                   # Исходные CSV, описание и рассчитанные признаки
├── models/                 # Модели, scaler, encoder, preprocessor и JSON
└── examples/               # EDA, ROC, матрицы, метрики и примеры
```

## Методические решения и ограничения

- Медианы Age по Pclass/Sex, мода порта и scaler обучены только по train split.
- Sex: female=0, male=1; инференс использует сохранённый LabelEncoder.
- Семейные признаки вычисляются одинаково для любых входных данных.
- Age_Group создан через pd.cut для EDA; в модель входит непрерывный Age.
- Модели сохранены после обучения на 712 строках; их метрики относятся именно к этим объектам.
- Выбор лидера по одному validation не даёт независимой оценки выбранного победителя.
- Синтетические профили из задания не имеют истинных меток; «ожидания» не подменяют прогноз.
- Готовые исходные submission.csv и gender_submission.csv не используются как разметка.

Источники и раскрытие использования ИИ приведены в разделах 1 и 13 ноутбука.
