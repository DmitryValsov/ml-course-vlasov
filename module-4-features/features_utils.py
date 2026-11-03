"""Воспроизводимая инженерия признаков Online Shoppers и сырой инференс."""

from pathlib import Path  # Переносимые пути внутри модуля.
from io import BytesIO  # Загрузка joblib из памяти.
import json  # Сохраняем схему и параметры.
import re  # Проверяем адрес собственного GitHub-репозитория.
import joblib  # Сериализация моделей и трансформеров.
import numpy as np  # Безопасные отношения и циклические признаки.
import pandas as pd  # Работа со смешанными таблицами.
import requests  # Реальная raw-загрузка после самостоятельной публикации.
from sklearn.base import BaseEstimator, TransformerMixin  # Совместимость с clone/Pipeline/CV.
from sklearn.preprocessing import LabelEncoder  # Обучаем категории только внутри fit.
from sklearn.utils.validation import check_is_fitted  # Проверка перед transform.

NUMERIC = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
CATEGORICAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
TECHNICAL = ["OperatingSystems", "Browser", "Region", "TrafficType"]  # Номинальные коды, не количественные шкалы.
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}  # В CSV используется June, не Jun.
ENGINEERED = ["TotalPages", "TotalDuration", "AvgPageValue", "BounceExitRatio", "IsNewVisitor", "IsWeekend",
              "MonthNum", "IsHighTraffic", "MonthSin", "MonthCos", "AvgDurationPerPage", "ProductPageShare"]


def _normal_categories(frame):
    result = frame.copy()  # Не меняем вход пользователя.
    for column in TECHNICAL:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("Int64").astype("string")  # 1 и 1.0 — одна категория.
    weekend = result["Weekend"].astype("string").str.lower()  # Поддерживаем bool и строковое представление JSON/CSV.
    result["Weekend"] = weekend.replace({"1": "true", "0": "false"})  # Не используем bool("False").
    result["Month"] = result["Month"].astype("string").replace({"Jun": "June"})  # Единый уровень июня.
    result["VisitorType"] = result["VisitorType"].astype("string")  # Явный тип категорий.
    return result


class ShopperPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, engineered=True, drop_page_values=False):
        self.engineered = engineered  # Позволяет сравнить исходную и расширенную схемы в CV.
        self.drop_page_values = drop_page_values  # Абляция PageValues и всех его производных.

    def fit(self, X, y=None):
        raw = pd.DataFrame(X).drop(columns="Revenue", errors="ignore").copy()  # Цель не используется в признаках.
        self.input_cols_ = [c for c in raw if c in NUMERIC + CATEGORICAL]  # Фиксируем порядок исходных 17 полей.
        if set(self.input_cols_) != set(NUMERIC + CATEGORICAL):
            raise ValueError("Для fit требуется исходная схема Online Shoppers из 17 полей")
        raw = _normal_categories(raw)  # Типы категорий одинаковы при обучении и инференсе.
        self.medians_ = {c: float(pd.to_numeric(raw[c], errors="coerce").median()) for c in NUMERIC}  # Только текущий train/CV-фолд.
        self.categorical_cols_ = list(CATEGORICAL)  # Имена категорий для нативного режима CatBoost.
        self.modes_, self.label_encoders_ = {}, {}  # Отдельное сохранённое соответствие для каждой категории.
        for column in CATEGORICAL:
            self.modes_[column] = str(raw[column].mode().iloc[0])  # Резервная категория для настоящего NA.
            self.label_encoders_[column] = LabelEncoder().fit(raw[column].fillna(self.modes_[column]).astype(str))
        columns = self.input_cols_ + (ENGINEERED if self.engineered else [])  # Инженерия меняет схему явно.
        self.feature_cols_ = [c for c in columns if not (self.drop_page_values and c in ["PageValues", "AvgPageValue"])]
        self.fit_rows_, self.n_features_in_ = len(raw), len(self.input_cols_)  # Аудит обучающей границы.
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_cols_")
        data = pd.DataFrame(X).reindex(columns=self.input_cols_).copy()  # Сохраняем заданную схему.
        data = _normal_categories(data)  # Не обучаем новые соответствия.
        for column in NUMERIC:
            data[column] = pd.to_numeric(data[column], errors="raise").fillna(self.medians_[column])  # Неизвестный текст не превращаем в число.
        for column in CATEGORICAL:
            data[column] = data[column].fillna(self.modes_[column])  # Берём моду обучения, а не test.
        if self.engineered:
            data["TotalPages"] = data[["Administrative", "Informational", "ProductRelated"]].sum(axis=1)  # Сумма просмотренных страниц.
            data["TotalDuration"] = data[["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]].sum(axis=1)
            data["AvgPageValue"] = data["PageValues"] / (data["TotalPages"] + 1)  # Формула задания; не новая независимая сумма выручки.
            data["BounceExitRatio"] = data["BounceRates"] / (data["ExitRates"] + .001)  # Защита нулевого ExitRates.
            data["IsNewVisitor"] = (data["VisitorType"] == "New_Visitor").astype(int)
            data["IsWeekend"] = (data["Weekend"] == "true").astype(int)  # Смысловой бинарный код.
            data["MonthNum"] = data["Month"].map(MONTHS)  # June не вызывает ошибку %b.
            if data["MonthNum"].isna().any():
                raise ValueError("Неизвестное название месяца")
            data["IsHighTraffic"] = (pd.to_numeric(data["TrafficType"]) > 10).astype(int)  # Порог кода, не объём трафика.
            angle = 2 * np.pi * (data["MonthNum"] - 1) / 12  # Замыкаем декабрь и январь в цикле.
            data["MonthSin"], data["MonthCos"] = np.sin(angle), np.cos(angle)
            data["AvgDurationPerPage"] = data["TotalDuration"] / (data["TotalPages"] + 1)  # Безопасное среднее времени.
            data["ProductPageShare"] = data["ProductRelated"] / (data["TotalPages"] + 1)  # Доля товарных страниц со сглаживанием.
        for column in CATEGORICAL:
            encoder = self.label_encoders_[column]  # Те же уровни, что при fit.
            mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            data[column] = data[column].astype(str).map(mapping).fillna(-1).astype(int)  # Неизвестная номинальная категория — код -1.
        encoded = data.loc[:, self.feature_cols_]  # Единственный порядок модели.
        if not np.isfinite(encoded.to_numpy(dtype=float)).all():
            raise ValueError("В подготовленных признаках есть NaN или бесконечность")
        return encoded

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_cols_")
        return np.array(self.feature_cols_, dtype=object)


def predict_session(session, model, preprocessor, feature_cols):
    if "Revenue" in session:
        raise ValueError("Удалите Revenue из входного профиля")
    missing = set(preprocessor.input_cols_) - set(session)
    if missing:
        raise ValueError(f"Отсутствуют поля: {sorted(missing)}")
    raw = pd.DataFrame([dict(session)])  # Не меняем входной словарь.
    for column in NUMERIC:
        if pd.isna(raw.at[0, column]):
            continue  # Настоящий пропуск обрабатывается сохранённой медианой.
        value = float(raw.at[0, column])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{column} должен быть конечным неотрицательным числом")
        if column in ["BounceRates", "ExitRates", "SpecialDay"] and value > 1:
            raise ValueError(f"{column} должен быть в диапазоне [0, 1]")
    features = preprocessor.transform(raw)  # Сырые поля проходят все обученные преобразования.
    if list(features.columns) != list(feature_cols):
        raise ValueError("Схема признаков не соответствует модели")
    probability = float(model.predict_proba(features)[0, 1])  # Модельная, не обязательно откалиброванная вероятность.
    return int(probability >= .5), probability


def load_local_artifacts(models_dir="models"):
    folder = Path(models_dir)
    keys = ["rf_model", "xgb_model", "lgbm_model", "cat_model", "tuned_lgbm", "best_model", "preprocessor", "scaler", "label_encoders"]
    loaded = {key: joblib.load(folder / f"{key}.pkl") for key in keys}
    loaded["feature_cols"] = json.loads((folder / "feature_cols.json").read_text(encoding="utf-8"))
    return loaded


def load_artifacts_from_github(repository_url, branch="main"):
    match = re.fullmatch(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?", repository_url.removesuffix(".git"))
    if not match:
        raise ValueError("Нужен адрес корня репозитория https://github.com/owner/repo")
    base = f"https://raw.githubusercontent.com/{match[1]}/{match[2]}/{branch}/module-4-features"
    loaded = {}
    for key in ["best_model", "preprocessor"]:
        response = requests.get(f"{base}/models/{key}.pkl", timeout=30)  # Только собственный доверенный репозиторий.
        response.raise_for_status()
        loaded[key] = joblib.load(BytesIO(response.content))
    response = requests.get(f"{base}/models/feature_cols.json", timeout=30)
    response.raise_for_status()
    loaded["feature_cols"] = response.json()
    return loaded
