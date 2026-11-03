"""Проверка сохранённого проекта без повторного обучения моделей."""

from pathlib import Path
from unittest.mock import patch
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix

from titanic_utils import fit_preprocessor, transform_passengers, predict_passenger
from titanic_utils import load_local_artifacts, interactive_predict


def verify():
    folder = Path(__file__).resolve().parent  # Проверка не зависит от каталога запуска.
    metadata = json.loads((folder / "models/metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((folder / "models/metrics.json").read_text(encoding="utf-8"))
    bundle = load_local_artifacts(folder / "models")  # Никаких вызовов fit моделей.
    raw = pd.read_csv(folder / "data/train.csv")
    test = pd.read_csv(folder / "data/test.csv")
    checks = []  # Сохраняем конкретный протокол вместо общей надписи о результате.

    for name in ["train.csv", "test.csv"]:
        actual_hash = hashlib.sha256((folder / "data" / name).read_bytes()).hexdigest()
        assert actual_hash == metadata["source_sha256"][name]
    checks.append("SHA-256 обоих исходных CSV совпадает с метаданными")

    train_ids = metadata["train_passenger_ids"]
    valid_ids = metadata["validation_passenger_ids"]
    assert len(train_ids) == 712 and len(valid_ids) == 179
    assert set(train_ids).isdisjoint(valid_ids)
    assert set(train_ids) | set(valid_ids) == set(raw["PassengerId"])
    indexed = raw.set_index("PassengerId", drop=False)  # Восстанавливаем порядок исходного split.
    train = indexed.loc[train_ids]
    valid = indexed.loc[valid_ids]
    checks.append("Разбиение 712/179 полно и не имеет пересечения")

    state = bundle["preprocessor"]
    independently_fitted = fit_preprocessor(train.drop(columns="Survived"))
    for name in ["age_medians", "age_global", "fare_median", "embarked_mode", "fit_rows"]:
        assert state[name] == independently_fitted[name]
    assert state["le_sex"].transform(["female", "male"]).tolist() == [0, 1]
    checks.append("Медианы, мода и кодировка сохранены по train split; female=0, male=1")

    _, train_features = transform_passengers(train, state, bundle["feature_cols"])
    _, valid_features = transform_passengers(valid, state, bundle["feature_cols"])
    _, test_features = transform_passengers(test, state, bundle["feature_cols"])
    np.testing.assert_allclose(bundle["scaler"].mean_, train_features.mean().to_numpy())
    np.testing.assert_allclose(bundle["scaler"].var_, train_features.var(ddof=0).to_numpy())
    assert "Survived" not in bundle["feature_cols"]
    assert test_features.shape == (418, 10)
    checks.append("Scaler соответствует train; в X нет Survived; Kaggle test имеет 418×10 признаков")

    key_map = {"logistic_regression": "lr_model", "decision_tree": "dt_model", "random_forest": "rf_model"}
    for metric_key, artifact_key in key_map.items():
        inputs = bundle["scaler"].transform(valid_features) if metric_key == "logistic_regression" else valid_features
        model = bundle[artifact_key]
        pred = model.predict(inputs)
        proba = model.predict_proba(inputs)[:, 1]
        expected = metrics[metric_key]
        calculated = {
            "accuracy": accuracy_score(valid["Survived"], pred),
            "precision": precision_score(valid["Survived"], pred, zero_division=0),
            "recall": recall_score(valid["Survived"], pred, zero_division=0),
            "f1": f1_score(valid["Survived"], pred, zero_division=0),
            "roc_auc": roc_auc_score(valid["Survived"], proba),
        }
        for name, value in calculated.items():
            np.testing.assert_allclose(value, expected[name], rtol=0, atol=1e-12)
        assert confusion_matrix(valid["Survived"], pred, labels=[0, 1]).tolist() == expected["confusion_matrix"]
    checks.append("Все пять метрик и матрицы ошибок трёх сохранённых моделей воспроизведены")

    passenger = {"Pclass": 1, "Sex": "female", "Age": 25, "SibSp": 1,
                 "Parch": 0, "Fare": 100, "Embarked": "C"}
    before = passenger.copy()
    lr = predict_passenger(passenger, bundle["lr_model"], bundle["scaler"],
                           bundle["feature_cols"], preprocessor=state)
    assert passenger == before  # Инференс не должен менять пользовательские данные.
    _, single = transform_passengers(pd.DataFrame([passenger]), state, bundle["feature_cols"])
    np.testing.assert_allclose(lr[1], bundle["lr_model"].predict_proba(bundle["scaler"].transform(single))[0, 1])
    missing = dict(passenger, Age=None, Fare=None, Embarked=None)
    _, missing_features = transform_passengers(pd.DataFrame([missing]), state, bundle["feature_cols"])
    assert np.isfinite(missing_features.to_numpy()).all()
    assert missing_features.iloc[0]["Age"] == state["age_medians"][(1, "female")]
    checks.append("Поштучный и пакетный инференс согласованы; пропуски заполняются сохранёнными значениями")

    invalid = [dict(passenger, Pclass=4), dict(passenger, Pclass=1.5),
               dict(passenger, Sex="unknown"), dict(passenger, Age=-1),
               dict(passenger, Age=float("inf")), dict(passenger, Embarked="X"),
               {k: v for k, v in passenger.items() if k != "SibSp"}]
    for sample in invalid:
        try:
            predict_passenger(sample, bundle["lr_model"], bundle["scaler"],
                              bundle["feature_cols"], preprocessor=state)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Некорректный профиль принят: {sample}")
    checks.append("Семь некорректных профилей отклонены с ValueError")

    with patch("builtins.input", side_effect=["1", "female", "25", "1", "0", "100", "C"]):
        interactive_result = interactive_predict(bundle["lr_model"], bundle["scaler"], bundle["feature_cols"], state)
    assert interactive_result == lr
    checks.append("Интерактивный ввод проверен подстановкой семи ответов")

    submission = pd.read_csv(folder / "submission.csv")
    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert submission["PassengerId"].tolist() == test["PassengerId"].tolist()
    assert len(submission) == 418 and submission["Survived"].isin([0, 1]).all()
    best_key = metadata["best_model_on_validation"]
    best_input = bundle["scaler"].transform(test_features) if best_key == "logistic_regression" else test_features
    assert submission["Survived"].tolist() == bundle[key_map[best_key]].predict(best_input).tolist()
    checks.append("Submission содержит 418 корректных прогнозов выбранной модели в исходном порядке ID")

    notebook = json.loads((folder / "notebook.ipynb").read_text(encoding="utf-8"))
    headings = [line for c in notebook["cells"] if c["cell_type"] == "markdown"
                for line in "".join(c["source"]).splitlines() if line.startswith("## ")]
    assert len(headings) == 14
    assert [int(h.split(". ")[0][3:]) for h in headings] == list(range(1, 15))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is not None
            assert not any(o["output_type"] == "error" for o in cell["outputs"])
    checks.append("Все 14 разделов идут по порядку; все кодовые ячейки выполнены без ошибок")
    for check in checks:
        print("OK:", check)
    print("GitHub raw: не проверен — публикацию автор выполняет самостоятельно.")
    return checks


if __name__ == "__main__":
    verify()
