"""Проверки сохранённых моделей и инженерии признаков без повторного обучения ансамблей."""

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
from features_utils import ShopperPreprocessor, ENGINEERED, predict_session, load_local_artifacts


def verify():
    folder = Path(__file__).resolve().parent
    metadata = json.loads((folder / "models/metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((folder / "models/metrics.json").read_text())
    cv = json.loads((folder / "models/cv_results.json").read_text())
    study = json.loads((folder / "models/optuna.json").read_text())
    ablation = json.loads((folder / "models/ablation.json").read_text())
    explanation_meta = json.loads((folder / "models/shap_details.json").read_text())
    bundle = load_local_artifacts(folder / "models")  # Обучения моделей здесь нет.
    pipeline = joblib.load(folder / "models/best_pipeline.pkl")
    raw = pd.read_csv(folder / "data/online_shoppers_intention.csv")
    checks = []

    assert hashlib.sha256((folder / "data/online_shoppers_intention.csv").read_bytes()).hexdigest() == metadata["source"]["sha256"]
    assert raw.shape == (12330, 18) and not raw.isna().any().any()
    duplicate_rows = raw.index[raw.duplicated()].tolist()
    assert len(duplicate_rows) == 125 and duplicate_rows == metadata["duplicate_rows_removed"]
    clean = raw.loc[~raw.duplicated()]
    train_rows, valid_rows = metadata["training_source_rows"], metadata["validation_source_rows"]
    assert len(clean) == 12205 and len(train_rows) == 9764 and len(valid_rows) == 2441
    assert set(train_rows).isdisjoint(valid_rows) and set(train_rows) | set(valid_rows) == set(clean.index)
    train = raw.loc[train_rows].drop(columns="Revenue")
    valid = raw.loc[valid_rows].drop(columns="Revenue")
    assert not set(pd.util.hash_pandas_object(train, index=False)).intersection(pd.util.hash_pandas_object(valid, index=False))
    y_valid = raw.loc[valid_rows, "Revenue"].astype(int).to_numpy()
    checks.append("Оригинал 12330×18 и SHA-256 проверены; удалены 125 повторов; 9764/2441 строк и профилей не пересекаются")

    preprocessor = bundle["preprocessor"]
    assert preprocessor.fit_rows_ == 9764 and "Revenue" not in preprocessor.input_cols_
    independent = ShopperPreprocessor().fit(train)  # Проверяем только train-статистики преобразований.
    assert independent.medians_ == preprocessor.medians_ and independent.modes_ == preprocessor.modes_
    assert len(independent.label_encoders_) == 7
    assert all(independent.label_encoders_[c].classes_.tolist() == preprocessor.label_encoders_[c].classes_.tolist()
               for c in independent.label_encoders_)
    X_train, X_valid = preprocessor.transform(train), preprocessor.transform(valid)
    assert X_train.shape == (9764, 29) and X_valid.shape == (2441, 29)
    assert set(ENGINEERED).issubset(X_train.columns) and len(ENGINEERED) == 12
    pd.testing.assert_frame_equal(X_valid, independent.transform(valid))
    np.testing.assert_allclose(bundle["scaler"].mean_, X_train.mean().to_numpy())
    assert metadata["scaler_used_by_models"] is False
    checks.append("29 входов, 12 новых признаков, 7 LabelEncoder и медианы воспроизведены только по train; scaler вспомогательный")

    sample = valid.iloc[:1].copy()
    sample["Month"] = "June"
    june = preprocessor.transform(sample)
    sample["Month"] = "Jun"
    pd.testing.assert_frame_equal(june, preprocessor.transform(sample))
    assert june["MonthNum"].iloc[0] == 6
    for value, expected in [(False, 0), ("False", 0), (True, 1), ("True", 1)]:
        sample["Weekend"] = value
        assert preprocessor.transform(sample)["IsWeekend"].iloc[0] == expected
    zero = valid.iloc[:1].copy()
    for column in ["Administrative", "Informational", "ProductRelated", "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "BounceRates", "ExitRates"]:
        zero[column] = 0
    zero_features = preprocessor.transform(zero)
    assert np.isfinite(zero_features.to_numpy(dtype=float)).all()
    assert zero_features["TotalPages"].iloc[0] == zero_features["TotalDuration"].iloc[0] == zero_features["BounceExitRatio"].iloc[0] == 0
    np.testing.assert_allclose(np.square(X_valid["MonthSin"]) + np.square(X_valid["MonthCos"]), 1, atol=1e-12)
    checks.append("Проверены June/Jun, строковые и булевы Weekend, нулевые знаменатели и циклические координаты")

    raw_schema = ShopperPreprocessor(engineered=False).fit(train)
    without_page = ShopperPreprocessor(drop_page_values=True).fit(train)
    assert raw_schema.transform(valid).shape[1] == 17
    assert without_page.transform(valid).shape[1] == 27
    assert not {"PageValues", "AvgPageValue"}.intersection(without_page.get_feature_names_out())
    for key, result in {**cv, **ablation}.items():
        assert len(result["folds"]) == 5
        np.testing.assert_allclose(np.mean(result["folds"]), result["mean"], rtol=0, atol=1e-12)
    checks.append("Абляция 17/29 признаков и вариант без PageValues/AvgPageValue проверены; CV содержит по пять фолдов")

    artifacts = {"random_forest": "rf_model", "xgboost": "xgb_model", "lightgbm": "lgbm_model", "catboost": "cat_model", "lightgbm_optuna": "tuned_lgbm"}
    predictions = pd.read_csv(folder / "examples/holdout_predictions.csv")
    assert predictions["source_row"].tolist() == valid_rows and predictions["actual"].tolist() == y_valid.tolist()
    for key, name in artifacts.items():
        probability = bundle[name].predict_proba(X_valid)[:, 1]
        predicted = (probability >= .5).astype(int)
        actual = {"accuracy": accuracy_score(y_valid, predicted), "precision": precision_score(y_valid, predicted, zero_division=0),
                  "recall": recall_score(y_valid, predicted, zero_division=0), "f1": f1_score(y_valid, predicted, zero_division=0),
                  "roc_auc": roc_auc_score(y_valid, probability), "average_precision": average_precision_score(y_valid, probability)}
        for metric, value in actual.items():
            np.testing.assert_allclose(value, metrics[key][metric], rtol=0, atol=1e-12)
        np.testing.assert_allclose(predictions[key], probability, rtol=1e-7, atol=1e-7)  # CSV для XGBoost имеет исходную точность float32.
        assert confusion_matrix(y_valid, predicted, labels=[0, 1]).tolist() == metrics[key]["confusion_matrix"]
    assert metadata["selected_model"] == max(cv, key=lambda key: cv[key]["mean"])
    checks.append("Все шесть метрик, матрицы и вероятности пяти моделей воспроизведены; выбор сделан по CV")

    trials = pd.read_csv(folder / "examples/optuna_trials.csv")
    assert len(trials) == study["n_trials"] == 30 and trials["state"].eq("COMPLETE").all()
    assert trials["number"].is_unique
    np.testing.assert_allclose(trials["value"].max(), study["best_cv_auc"], atol=1e-12, rtol=0)
    checks.append("30 завершённых испытаний Optuna подтверждены сохранённой историей")

    sessions = json.loads((folder / "examples/demo_sessions.json").read_text())
    result = []
    for session in sessions:
        before = copy.deepcopy(session)
        prediction = predict_session(session, bundle["best_model"], preprocessor, bundle["feature_cols"])
        assert session == before and 0 <= prediction[1] <= 1
        result.append(prediction[1])
    np.testing.assert_allclose(result, pipeline.predict_proba(pd.DataFrame(sessions))[:, 1], rtol=0, atol=1e-12)
    invalid = [dict(sessions[0], Revenue=True), dict(sessions[0], ProductRelated=-1),
               dict(sessions[0], ExitRates=1.5), dict(sessions[0], PageValues=float("inf")),
               dict(sessions[0], Month="InvalidMonth"), {k: v for k, v in sessions[0].items() if k != "VisitorType"}]
    for item in invalid:
        try:
            predict_session(item, bundle["best_model"], preprocessor, bundle["feature_cols"])
        except ValueError:
            pass
        else:
            raise AssertionError("Некорректный профиль принят")
    unknown = dict(sessions[0], Browser=999)
    assert preprocessor.transform(pd.DataFrame([unknown]))["Browser"].iloc[0] == -1
    checks.append("Три сырых прогноза согласованы с Pipeline; словари сохраняются; шесть ошибочных входов отклонены, новый браузер обработан")

    explanation = np.load(folder / "examples/shap_values.npz", allow_pickle=False)
    positions = explanation["positions"]
    assert explanation["values"].shape == (len(positions), 29)
    np.testing.assert_allclose(explanation["data"], X_valid.iloc[positions].to_numpy())
    p = bundle["best_model"].predict_proba(X_valid.iloc[positions])[:, 1]
    expected = p if explanation_meta["units"] == "probability" else np.log(np.clip(p, 1e-12, 1-1e-12) / np.clip(1-p, 1e-12, 1))
    np.testing.assert_allclose(explanation["base_values"] + explanation["values"].sum(axis=1), expected, rtol=1e-4, atol=1e-4)
    assert explanation_meta["risk_position"] in positions
    assert explanation_meta["source_row"] == valid_rows[explanation_meta["risk_position"]]
    checks.append("SHAP соответствует holdout и исходным строкам; аддитивность raw-выхода подтверждена")

    notebook = json.loads((folder / "notebook.ipynb").read_text(encoding="utf-8"))
    headers = [line for cell in notebook["cells"] if cell["cell_type"] == "markdown"
               for line in "".join(cell["source"]).splitlines() if line.startswith("## ")]
    assert len(headers) == 14 and [int(h.split(". ")[0][3:]) for h in headers] == list(range(1, 15))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is not None and not any(o["output_type"] == "error" for o in cell["outputs"])
            ast.parse("".join(cell["source"]))
    for source in folder.glob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"))
    assert metadata["teacher"] == "Спирин Игорь Сергеевич"
    checks.append("14 разделов по порядку, все ячейки выполнены, Python-синтаксис и полное ФИО проверены")
    for check in checks:
        print("OK:", check)
    print("GitHub raw пока не проверен: автор публикует самостоятельно.")
    return checks


if __name__ == "__main__":
    verify()
