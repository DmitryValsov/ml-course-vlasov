# Модуль 4 — инженерия признаков Online Shoppers

**Автор:** Власов Дмитрий Сергеевич, АСОиУб-23-2  
**Преподаватель:** Спирин Игорь Сергеевич  
**Дата выполнения:** 03.11.2026

## Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | CV AUC | Fit, с |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8849 | 0.6020 | 0.7801 | 0.6796 | 0.9327 | 0.7342 | 0.9316 | 0.8518 |
| XGBoost | 0.8894 | 0.6223 | 0.7461 | 0.6786 | 0.9281 | 0.7333 | 0.9242 | 0.2902 |
| LightGBM | 0.8796 | 0.5846 | 0.7958 | 0.6741 | 0.9313 | 0.7394 | 0.9269 | 0.4036 |
| CatBoost | 0.8697 | 0.5565 | 0.8246 | 0.6646 | 0.9309 | 0.7323 | 0.9279 | 1.9273 |
| LightGBM (Optuna) | 0.8689 | 0.5527 | 0.8508 | 0.6701 | 0.9367 | 0.7542 | 0.9341 | 0.2427 |


По train CV выбрана **LightGBM (Optuna)**, CV AUC 0.9341.
Стратифицированное разбиение seed=42: 9764 train / 2441 holdout после удаления
125 точных повторов из исходных 12330 строк. Сырые CSV не изменены.
Все обучаемые преобразования находятся внутри CV Pipeline.

## Инженерия признаков

Созданы 12 признаков: восемь обязательных из задания и MonthSin, MonthCos,
AvgDurationPerPage, ProductPageShare. Исходных входов 17, полная схема — 29.
Одинаковый LightGBM на пяти CV-фолдах:

| Схема | CV AUC | Std фолдов |
|---|---:|---:|
| raw_17 | 0.9241 | 0.0067 |
| engineered_29 | 0.9269 | 0.0056 |
| without_page_values | 0.7689 | 0.0157 |


Изменение CV AUC от инженерии: +0.0028. Вариант без PageValues
также исключает AvgPageValue. Benchmark по агрегатам сессии не подтверждает
качество раннего прогноза; время расчёта PageValues требуется проверять.
IsHighTraffic — порог кода TrafficType, а не измеренная интенсивность.

## Просмотр и запуск

`notebook.ipynb` содержит 14 разделов с выполненными ячейками; `report.html`
показывает работу с встроенными графиками без Python. Для расчёта:

```bash
cd module-4-features
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebook.ipynb
```

На macOS нужна OpenMP (`brew install libomp` либо conda-forge llvm-openmp).
`python full_code.py` повторяет весь расчёт с CV, абляциями и 30 пробами Optuna.
`python verify_results.py` проверяет готовые артефакты без повторного обучения моделей.

## Содержимое

- data/: оригинальный CSV UCI, CC BY 4.0, SHA-256, описание всех полей.
- models/: baseline, tuned LightGBM, best_model, Pipeline, preprocessor, кодировщики,
  вспомогательный scaler, CV, абляции, Optuna, метрики и SHAP-метаданные.
- examples/: EDA, ROC, матрицы, importance, абляция, история Optuna, четыре графика SHAP,
  массив SHAP, holdout-прогнозы и три сырых примера.
- features_utils.py, full_code.py: полный комментированный код.

## Сырой инференс

```python
import json
from pathlib import Path
from features_utils import load_local_artifacts, predict_session

bundle = load_local_artifacts("models")
session = json.loads(Path("examples/demo_sessions.json").read_text())[0]
label, probability = predict_session(session, bundle["best_model"],
    bundle["preprocessor"], bundle["feature_cols"])
print(label, probability)
```

Нужен полный сырой профиль из 17 полей без Revenue. Модели используют
немасштабированные признаки; scaler сохранён по структуре задания и не включён
в инференс. Балансировочные веса не гарантируют калибровку вероятностей.
Для joblib-трансформера нужен features_utils.py в пути импорта.

## Публикация

Автор публикует модуль в общем ml-course-vlasov рядом с модулями 1–3.
После загрузки задайте repository_url и run_github_check=true в student.json,
выполните раздел 12 и обновите фактический статус. Локальные три класса и
вероятности совпали; настоящая GitHub raw-загрузка пока не выполнялась.
Подробная инструкция — в корневом GITHUB_UPLOAD.md.
