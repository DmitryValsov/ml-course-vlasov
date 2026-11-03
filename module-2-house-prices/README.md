# Модуль 2 — House Prices: предобработка и регрессия

**Автор:** Власов Дмитрий Сергеевич, АСОиУб-23-2  
**Преподаватель:** Спирин Игорь Сергеевич  
**Дата выполнения и обучения:** 15.10.2026

## Результаты на 292 отложенных домах

| Модель | RMSE log | MAE log | R² log | RMSE USD | MAE USD | R² USD |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.1233 | 0.0867 | 0.9185 | 22,339 | 15,027 | 0.9349 |
| Ridge | 0.1348 | 0.0892 | 0.9026 | 24,248 | 15,400 | 0.9233 |
| Lasso | 0.1334 | 0.0867 | 0.9047 | 24,335 | 14,971 | 0.9228 |
| ElasticNet | 0.1336 | 0.0869 | 0.9043 | 24,179 | 14,986 | 0.9238 |
| Gradient Boosting | 0.1352 | 0.0881 | 0.9020 | 27,277 | 15,793 | 0.9030 |


По минимальному train CV RMSE(log) выбрана **Lasso**:
CV RMSE(log) 0.1110.
Выбор выполнен до оценки holdout. Подбор alpha/l1_ratio — GridSearchCV, 5 фолдов,
вся предобработка внутри Pipeline. Внешний split: 1168/292, random_state=42;
из train удалены только две предметные аномалии (Id 524, 1299), осталось 1166 строк.

R²(log) и R²(USD) относятся к разным шкалам. Уровень 0.90
достигнут в log-шкале и
достигнут в USD у выбранной модели.
Порог не подгонялся фильтрацией holdout. Время финального fit выбранного Pipeline:
0.2292 с, полного CV+fit: 12.86 с.

## Просмотр и запуск

`notebook.ipynb` — основной отчёт из 14 разделов с выполненными ячейками.
`report.html` — та же работа для браузера, графики встроены в файл.
Для повторного расчёта рекомендуется Python 3.12:

```bash
cd module-2-house-prices
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebook.ipynb
```

Альтернатива: `python full_code.py`. Проверка готовых артефактов без переобучения:
`python verify_results.py`. При повторном запуске время fit изменится.


## 📁 Структура модуля

```
module-2-house-prices/
├── README.md                    ← этот файл
├── notebook.ipynb               ← полный отчёт: 14 разделов
├── requirements.txt             ← зависимости
├── data/
│   ├── train.csv                ← 1460 строк с целевой переменной
│   ├── test.csv                 ← 1459 строк для submission
│   └── house_prices_info.md     ← описание датасета
└── models/                      ← сохранённые модели и метаданные
    ├── lr_model.pkl             ← Linear Regression
    ├── ridge_model.pkl          ← Ridge
    ├── lasso_model.pkl          ← Lasso (лучшая модель)
    ├── enet_model.pkl           ← ElasticNet
    ├── scaler.pkl               ← StandardScaler
    ├── feature_cols.json        ← список признаков
    └── metrics.json             ← метрики всех моделей
```


## Быстрый локальный прогноз

```python
import json
from pathlib import Path
from house_utils import load_local_artifacts, predict_price

bundle = load_local_artifacts("models")
house = json.loads(Path("examples/demo_houses.json").read_text())[1]
price = predict_price(house, bundle["best_model"], bundle["scaler"],
                      bundle["feature_cols"], bundle["preprocessor"])
print(f"${price:,.0f}")
```

Для joblib-десериализации `HousePreprocessor` требуется `house_utils.py`
в пути импорта. Загружайте только собственные доверенные модели.
Отсутствующие поля сырого дома заполняются параметрами обучения, а не нулями
для всех признаков. Для осмысленного прогноза нужен полный профиль.

## Файлы

- `data/train.csv`, `test.csv` — учебная копия Kaggle из D2L; источник и SHA-256 в `data/sources.json`.
- `data/house_prices_info.md` — схема всех полей, пропуски и статистики цены.
- `models/` — четыре обязательных модели, бустинг, выбранная модель, полный Pipeline, scaler, preprocessor и JSON.
- `examples/` — EDA, остатки, предсказания, коэффициенты, таблица метрик, CV и пять примеров.

Цены являются историческими USD Ames, площади — sq ft. Современный рынок,
инфляция и интервалы прогноза не моделируются. Готовые ожидания синтетических
примеров не являются истинными ценами. Модуль 1 хранится рядом и не переделывался.
