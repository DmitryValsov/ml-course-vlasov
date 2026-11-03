"""Проверка сохранённых моделей, Optuna, SHAP и отчёта без переобучения моделей."""

from pathlib import Path
import ast
import copy
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from churn_utils import ChurnPreprocessor, predict_customer, load_local_artifacts


def verify():
    folder = Path(__file__).resolve().parent
    metadata = json.loads((folder / "models/metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((folder / "models/metrics.json").read_text())
    cv = json.loads((folder / "models/cv_results.json").read_text())
    study = json.loads((folder / "models/optuna.json").read_text())
    shap_info = json.loads((folder / "models/shap_details.json").read_text())
    bundle = load_local_artifacts(folder / "models")  # Только загрузка готовых объектов.
    pipeline = joblib.load(folder / "models/best_pipeline.pkl")
    raw = pd.read_csv(folder / "data/telco_churn.csv")
    indexed = raw.set_index("customerID", drop=False)
    train_ids, valid_ids = metadata["training_customer_ids"], metadata["validation_customer_ids"]
    checks = []

    assert hashlib.sha256((folder / "data/telco_churn.csv").read_bytes()).hexdigest() == metadata["source"]["sha256"]
    assert raw.shape == (7043, 21) and raw["customerID"].is_unique
    assert len(train_ids) == 5634 and len(valid_ids) == 1409
    assert set(train_ids).isdisjoint(valid_ids) and set(train_ids) | set(valid_ids) == set(raw["customerID"])
    checks.append("SHA-256, размер 7043×21 и непересекающееся разбиение 5634/1409 проверены")

    train = indexed.loc[train_ids].drop(columns=["customerID", "Churn"])
    valid = indexed.loc[valid_ids].drop(columns=["customerID", "Churn"])
    y_valid = indexed.loc[valid_ids, "Churn"].map({"No": 0, "Yes": 1}).to_numpy()
    preprocessor = bundle["preprocessor"]
    assert preprocessor.fit_rows_ == 5634
    assert not {"customerID", "Churn"}.intersection(preprocessor.input_cols_)
    independent = ChurnPreprocessor().fit(train)  # Проверяем обученные статистики, не регрессоры/классификаторы.
    assert independent.numeric_medians_ == preprocessor.numeric_medians_
    assert all(independent.label_encoders_[c].classes_.tolist() == preprocessor.label_encoders_[c].classes_.tolist()
               for c in preprocessor.categorical_cols_)
    X_train, X_valid = preprocessor.transform(train), preprocessor.transform(valid)
    assert X_train.shape == (5634, 23) and X_valid.shape == (1409, 23)
    assert list(X_valid.columns) == bundle["feature_cols"]
    np.testing.assert_allclose(bundle["scaler"].mean_, X_train.mean().to_numpy())
    assert metadata["scaler_used_by_models"] is False
    checks.append("23 признака, train-медианы и 16 LabelEncoder воспроизведены; scaler не используется моделями")

    artifacts = {"random_forest": "rf_model", "xgboost": "xgb_model", "lightgbm": "lgbm_model",
                 "catboost": "cat_model", "lightgbm_optuna": "tuned_lgbm"}
    saved_predictions = pd.read_csv(folder / "examples/holdout_predictions.csv")
    assert saved_predictions["customerID"].tolist() == valid_ids
    assert saved_predictions["actual"].tolist() == y_valid.tolist()
    for key, name in artifacts.items():
        probability = bundle[name].predict_proba(X_valid)[:, 1]
        predicted = (probability >= .5).astype(int)
        actual = {"accuracy": accuracy_score(y_valid, predicted), "precision": precision_score(y_valid, predicted, zero_division=0),
                  "recall": recall_score(y_valid, predicted, zero_division=0), "f1": f1_score(y_valid, predicted, zero_division=0),
                  "roc_auc": roc_auc_score(y_valid, probability), "average_precision": average_precision_score(y_valid, probability)}
        for metric, value in actual.items():
            np.testing.assert_allclose(value, metrics[key][metric], rtol=0, atol=1e-12)
        np.testing.assert_allclose(saved_predictions[key], probability, rtol=1e-7, atol=1e-7)  # CSV экспортирует float32 XGBoost с его исходной точностью.
        assert confusion_matrix(y_valid, predicted, labels=[0, 1]).tolist() == metrics[key]["confusion_matrix"]
    assert metadata["selected_model"] == max(cv, key=lambda k: cv[k]["mean"])
    checks.append("Шесть метрик, матрицы и вероятности всех пяти сохранённых моделей воспроизведены; итог выбран по CV")

    trials = pd.read_csv(folder / "examples/optuna_trials.csv")
    assert len(trials) == study["n_trials"] == 30 and trials["state"].eq("COMPLETE").all()
    assert trials["number"].is_unique and trials["value"].notna().all()
    np.testing.assert_allclose(trials["value"].max(), study["best_cv_auc"], rtol=0, atol=1e-12)
    for result in cv.values():
        assert len(result["folds"]) == 5
        np.testing.assert_allclose(np.mean(result["folds"]), result["mean"], rtol=0, atol=1e-12)
    checks.append("30 завершённых испытаний Optuna и по пять CV-оценок каждого кандидата подтверждены")

    clients = json.loads((folder / "examples/demo_clients.json").read_text())
    results = []
    for client in clients:
        before = copy.deepcopy(client)
        result = predict_customer(client, bundle["best_model"], preprocessor, bundle["feature_cols"])
        assert client == before and 0 <= result[1] <= 1
        results.append(result)
    np.testing.assert_allclose([r[1] for r in results], pipeline.predict_proba(pd.DataFrame(clients))[:, 1], rtol=0, atol=1e-12)
    unknown = dict(clients[0], PaymentMethod="NEW_PAYMENT")
    assert preprocessor.transform(pd.DataFrame([unknown]))["PaymentMethod"].iloc[0] == -1
    missing = dict(clients[0], TotalCharges=" ")
    assert np.isfinite(preprocessor.transform(pd.DataFrame([missing])).to_numpy(dtype=float)).all()
    invalid = [dict(clients[0], tenure=-1), dict(clients[0], MonthlyCharges=float("inf")),
               dict(clients[0], Churn="Yes"), {k: v for k, v in clients[0].items() if k != "Contract"}]
    for sample in invalid:
        try:
            predict_customer(sample, bundle["best_model"], preprocessor, bundle["feature_cols"])
        except ValueError:
            pass
        else:
            raise AssertionError("Некорректный профиль принят")
    checks.append("Три индивидуальных прогноза согласованы с Pipeline; проверены пробел TotalCharges, новый уровень и ошибки ввода")

    explanation = np.load(folder / "examples/shap_values.npz", allow_pickle=False)
    positions = explanation["positions"]
    assert len(positions) == shap_info["sample_rows"] and explanation["values"].shape == (len(positions), 23)
    np.testing.assert_allclose(explanation["data"], X_valid.iloc[positions].to_numpy())
    p = bundle["best_model"].predict_proba(X_valid.iloc[positions])[:, 1]
    expected = p if shap_info["units"] == "probability" else np.log(np.clip(p, 1e-12, 1-1e-12) / np.clip(1-p, 1e-12, 1))
    np.testing.assert_allclose(explanation["base_values"] + explanation["values"].sum(axis=1), expected, rtol=1e-4, atol=1e-4)
    assert shap_info["risk_position"] in positions
    checks.append("SHAP-данные соответствуют holdout; аддитивность raw-выхода и включение высокого риска проверены")

    notebook = json.loads((folder / "notebook.ipynb").read_text(encoding="utf-8"))
    headings = [line for cell in notebook["cells"] if cell["cell_type"] == "markdown"
                for line in "".join(cell["source"]).splitlines() if line.startswith("## ")]
    assert len(headings) == 14 and [int(h.split(". ")[0][3:]) for h in headings] == list(range(1, 15))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is not None and not any(o["output_type"] == "error" for o in cell["outputs"])
            ast.parse("".join(cell["source"]))
    for file in folder.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"))
    assert metadata["teacher"] == "Спирин Игорь Сергеевич"
    checks.append("14 разделов, выполненные ячейки, Python-синтаксис и полное ФИО преподавателя проверены")
    for message in checks:
        print("OK:", message)
    print("GitHub raw не проверен: публикация выполняется автором самостоятельно.")
    return checks


if __name__ == "__main__":
    verify()
