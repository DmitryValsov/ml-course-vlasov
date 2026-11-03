# Модуль 3 — ансамбли и Telco Customer Churn

**Автор:** Власов Дмитрий Сергеевич, АСОиУб-23-2  
**Преподаватель:** Спирин Игорь Сергеевич  
**Дата выполнения:** 03.11.2026

## Результаты на 1409 отложенных клиентах

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | CV AUC | Fit, с |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.7764 | 0.5616 | 0.7193 | 0.6307 | 0.8414 | 0.8438 | 0.4141 |
| XGBoost | 0.7551 | 0.5294 | 0.6979 | 0.6021 | 0.8281 | 0.8319 | 0.1720 |
| LightGBM | 0.7573 | 0.5305 | 0.7433 | 0.6192 | 0.8366 | 0.8405 | 0.3092 |
| CatBoost | 0.7601 | 0.5332 | 0.7727 | 0.6310 | 0.8446 | 0.8438 | 1.5304 |
| LightGBM (Optuna) | 0.7438 | 0.5111 | 0.8021 | 0.6243 | 0.8474 | 0.8490 | 0.2176 |


Итог **LightGBM (Optuna)** выбран по train CV AUC 0.8490,
без использования holdout при подборе. Разбиение стратифицировано, seed=42:
5634 train / 1409 holdout. Optuna: 30 завершённых испытаний × 5 фолдов,
полное время поиска 38.41 с. Изменение holdout AUC
LightGBM после поиска: +0.0108. Подробности и ограничения приведены в ноутбуке.

## Просмотр и повторный запуск

- `notebook.ipynb` — 14 разделов, все ячейки выполнены.
- `report.html` — готовая версия с встроенными графиками для браузера.
- `full_code.py` — весь сценарий с маркерами `# %%`.
- `churn_utils.py` — переносимая предобработка и инференс.
- `verify_results.py` — проверка сохранённых файлов без переобучения моделей.

Рекомендуется Python 3.12 в отдельном окружении. На macOS нужна OpenMP:
`brew install libomp` либо conda-forge `llvm-openmp`. На старом macOS готовая
библиотека conda-forge может быть удобнее сборки Homebrew из исходников.

```bash
cd module-3-churn
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebook.ipynb
```

Альтернатива: `python full_code.py`. Полный поиск занимает несколько минут.
Для проверки готовых артефактов: `python verify_results.py`.

## Что сохранено

`data/`: исходный CSV IBM, источник с SHA-256, описание всех 21 столбца.
`models/`: четыре baseline, tuned_lgbm, best_model, полный Pipeline, preprocessor,
LabelEncoder, вспомогательный scaler, параметры Optuna, CV, метрики и SHAP-метаданные.
`examples/`: EDA, ROC, матрицы, importance, история 30 проб, четыре графика SHAP,
проверяемые SHAP-значения, holdout-прогнозы и три сырых клиента для инференса.

## Быстрый локальный инференс

```python
import json
from pathlib import Path
from churn_utils import load_local_artifacts, predict_customer

bundle = load_local_artifacts("models")
client = json.loads(Path("examples/demo_clients.json").read_text())[0]
label, probability = predict_customer(client, bundle["best_model"],
    bundle["preprocessor"], bundle["feature_cols"])
print(label, probability)
```

Нужен полный сырой профиль из 19 полей; customerID можно передать дополнительно,
Churn передавать запрещено. Порог — 0.5. Деревья обучены без масштабирования;
scaler.pkl сохранён по структуре задания и в прогнозах не участвует.
Калибровка не выполнялась: балансировочные веса влияют на смысл вероятности.
SHAP объясняет raw-выход, у бустинга — log-odds, и не доказывает причинность.

## GitHub

Публикацию автор выполняет самостоятельно в общем ml-course-vlasov.
После загрузки укажите repository_url и run_github_check=true в student.json,
запустите раздел 12 и сохраните реальный результат. Для импорта preprocessor
рядом нужен churn_utils.py. Локально три класса и вероятности совпадают;
настоящая raw-загрузка пока не выполнена. Инструкция — в корневом GITHUB_UPLOAD.md.
