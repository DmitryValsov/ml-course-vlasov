"""Проверяем сохранённые модели House Prices без повторного обучения регрессоров."""

from pathlib import Path
from unittest.mock import patch
import ast
import copy
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from house_utils import HousePreprocessor, predict_price, load_local_artifacts, interactive_predict


def verify():
    folder = Path(__file__).resolve().parent  # Не зависим от текущего каталога запуска.
    metadata = json.loads((folder / "models/metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((folder / "models/metrics.json").read_text(encoding="utf-8"))
    cv = json.loads((folder / "models/cv_results.json").read_text(encoding="utf-8"))
    sources = json.loads((folder / "data/sources.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(folder / "data/train.csv")
    test = pd.read_csv(folder / "data/test.csv")
    bundle = load_local_artifacts(folder / "models")  # Загружаем готовые объекты, не вызываем fit моделей.
    pipeline = joblib.load(folder / "models/best_pipeline.pkl")
    checks = []

    for name in ["train.csv", "test.csv"]:
        actual = hashlib.sha256((folder / "data" / name).read_bytes()).hexdigest()
        assert actual == sources["files"][name]["sha256"]
    assert raw.shape == (1460, 81) and test.shape == (1459, 80)
    assert "SalePrice" not in test and raw["Id"].is_unique and test["Id"].is_unique
    checks.append("SHA-256, размеры 1460×81 / 1459×80, ID и отсутствие разметки test проверены")

    train_ids, valid_ids = metadata["training_ids"], metadata["validation_ids"]
    removed = metadata["removed_ids"]
    assert len(train_ids) == 1166 and len(valid_ids) == 292 and set(removed) == {524, 1299}
    assert set(train_ids).isdisjoint(valid_ids)
    assert set(train_ids) | set(valid_ids) | set(removed) == set(raw["Id"])
    assert set(removed).issubset(metadata["outer_train_ids"])
    checks.append("1166 обучающих и 292 отложенных дома не пересекаются; удалены только два train-объекта")

    indexed = raw.set_index("Id", drop=False)
    train = indexed.loc[train_ids].drop(columns=["Id", "SalePrice"])
    valid = indexed.loc[valid_ids].drop(columns=["Id", "SalePrice"])
    actual_real = indexed.loc[valid_ids, "SalePrice"].to_numpy(dtype=float)
    actual_log = np.log1p(actual_real)
    preprocessor = bundle["preprocessor"]
    assert preprocessor.fit_rows_ == 1166
    assert not {"Id", "SalePrice", "SalePrice_log"}.intersection(preprocessor.input_cols_)
    encoded_train = preprocessor.transform(train)
    encoded_valid = preprocessor.transform(valid)
    encoded_test = preprocessor.transform(test.drop(columns="Id"))
    assert list(encoded_train.columns) == bundle["feature_cols"]
    assert encoded_train.shape == (1166, 244) and encoded_test.shape == (1459, 244)
    np.testing.assert_allclose(bundle["scaler"].mean_, encoded_train.mean().to_numpy(), rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(bundle["scaler"].var_, encoded_train.var(ddof=0).to_numpy(), rtol=1e-12, atol=1e-8)
    independent = clone(preprocessor).fit(train)  # Проверяем воспроизводимость только преобразований.
    assert independent.frontage_medians_ == preprocessor.frontage_medians_
    assert independent.categories_ == preprocessor.categories_ and independent.log_cols_ == preprocessor.log_cols_
    pd.testing.assert_frame_equal(independent.transform(valid), encoded_valid)
    checks.append("244 признака, медианы, категории и scaler воспроизводятся по train; цели и ID в X нет")

    mapping = {"linear_regression": "lr_model", "ridge": "ridge_model", "lasso": "lasso_model",
               "elasticnet": "enet_model", "gradient_boosting": "gbr_model"}
    scaled = bundle["scaler"].transform(encoded_valid)
    for key, artifact in mapping.items():
        pred_log = bundle[artifact].predict(scaled)
        pred_real = np.expm1(pred_log)
        recomputed = {"rmse_log": np.sqrt(mean_squared_error(actual_log, pred_log)),
                      "mae_log": mean_absolute_error(actual_log, pred_log), "r2_log": r2_score(actual_log, pred_log),
                      "rmse_real": np.sqrt(mean_squared_error(actual_real, pred_real)),
                      "mae_real": mean_absolute_error(actual_real, pred_real), "r2_real": r2_score(actual_real, pred_real)}
        for name, value in recomputed.items():
            np.testing.assert_allclose(value, metrics[key][name], rtol=1e-11, atol=1e-8)
    assert metadata["selected_model"] == min(cv, key=lambda key: cv[key]["rmse_log_mean"])
    assert not any(w["category"] == "ConvergenceWarning" for w in metadata["warnings"])
    checks.append("Все шесть holdout-метрик пяти моделей воспроизведены; выбор соответствует CV; сходимость подтверждена")

    houses = json.loads((folder / "examples/demo_houses.json").read_text(encoding="utf-8"))
    demo = pd.read_csv(folder / "examples/prediction_examples.csv")
    names = {"linear_regression": "Linear Regression", "ridge": "Ridge", "lasso": "Lasso",
             "elasticnet": "ElasticNet", "gradient_boosting": "Gradient Boosting"}
    for number, house in enumerate(houses):
        before = copy.deepcopy(house)
        for key, artifact in mapping.items():
            price = predict_price(house, bundle[artifact], bundle["scaler"], bundle["feature_cols"], preprocessor)
            np.testing.assert_allclose(price, demo.iloc[number][names[key]], rtol=1e-12, atol=1e-7)
        assert house == before
        amount = house["TotalBsmtSF"] + house["1stFlrSF"] + house["2ndFlrSF"]
        bath = house["FullBath"] + .5 * house["HalfBath"] + house["BsmtFullBath"] + .5 * house["BsmtHalfBath"]
        assert amount == demo.iloc[number]["TotalSF"] and bath == demo.iloc[number]["TotalBath"]
    individual = [predict_price(h, bundle["best_model"], bundle["scaler"], bundle["feature_cols"], preprocessor) for h in houses]
    np.testing.assert_allclose(individual, np.expm1(pipeline.predict(pd.DataFrame(houses))), rtol=1e-12, atol=1e-7)
    checks.append("25 демонстрационных прогнозов воспроизведены; суммы площадей/ванных верны; словари не меняются; Pipeline согласован")

    partial = {"OverallQual": 6, "GrLivArea": 1500, "YearBuilt": 2000, "YrSold": 2010, "GarageCars": 2}
    partial_price = predict_price(partial, bundle["best_model"], bundle["scaler"], bundle["feature_cols"], preprocessor)
    assert np.isfinite(partial_price) and partial_price > 0
    unknown = dict(houses[1], Neighborhood="UNSEEN_NEIGHBORHOOD", LotFrontage=None)
    assert np.isfinite(preprocessor.transform(pd.DataFrame([unknown])).to_numpy()).all()
    invalid = [dict(partial, OverallQual=0), dict(partial, OverallQual=2.5), dict(partial, GrLivArea=0),
               dict(partial, GrLivArea=float("inf")), dict(partial, YearBuilt=2020),
               {k: v for k, v in partial.items() if k != "OverallQual"}, dict(partial, SalePrice=123456)]
    for house in invalid:
        try:
            predict_price(house, bundle["best_model"], bundle["scaler"], bundle["feature_cols"], preprocessor)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Некорректный профиль принят: {house}")
    with patch("builtins.input", side_effect=["6", "1500", "2000", "2010", "2"]):
        price_from_input = interactive_predict(bundle)
    np.testing.assert_allclose(price_from_input, partial_price, rtol=0, atol=1e-8)
    checks.append("Проверены неполный профиль, новый район, семь ошибочных входов и интерактивный режим")

    submission = pd.read_csv(folder / "submission.csv")
    assert submission.shape == (1459, 2) and list(submission.columns) == ["Id", "SalePrice"]
    assert submission["Id"].tolist() == test["Id"].tolist()
    assert submission["SalePrice"].gt(0).all() and np.isfinite(submission["SalePrice"]).all()
    submission_expected = np.expm1(pipeline.predict(test.drop(columns="Id")))
    np.testing.assert_allclose(submission["SalePrice"], submission_expected)
    checks.append("1459 прогнозов submission положительны, конечны и соответствуют выбранному Pipeline и порядку Id")

    notebook = json.loads((folder / "notebook.ipynb").read_text(encoding="utf-8"))
    headings = [line for c in notebook["cells"] if c["cell_type"] == "markdown"
                for line in "".join(c["source"]).splitlines() if line.startswith("## ")]
    assert len(headings) == 14 and [int(h.split(". ")[0][3:]) for h in headings] == list(range(1, 15))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is not None
            assert not any(output["output_type"] == "error" for output in cell["outputs"])
            ast.parse("".join(cell["source"]))
    for file in folder.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"))
    checks.append("14 разделов идут по порядку; все кодовые ячейки выполнены без ошибок; Python-синтаксис проверен")
    for message in checks:
        print("OK:", message)
    print("GitHub raw не проверен: публикация выполняется автором самостоятельно.")
    return checks


if __name__ == "__main__":
    verify()
