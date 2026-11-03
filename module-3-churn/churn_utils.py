"""Сериализуемая предобработка Telco Churn и инференс без повторного fit."""

from io import BytesIO  # Загружаем собственные joblib-файлы из памяти.
from pathlib import Path  # Пути относительно папки модуля.
import json  # Схема признаков сохраняется отдельно.
import re  # Проверяем адрес репозитория.
import joblib  # Восстанавливаем модели и трансформеры.
import numpy as np  # Проверяем численные значения.
import pandas as pd  # Работаем с сырыми и подготовленными таблицами.
import requests  # Выполняем настоящую загрузку GitHub raw после публикации.
from sklearn.base import BaseEstimator, TransformerMixin  # Совместимость с Pipeline и CV.
from sklearn.preprocessing import LabelEncoder  # Требуемое кодирование категорий.
from sklearn.utils.validation import check_is_fitted  # Не допускаем transform до обучения.

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]  # Три непрерывных поля исходного набора.
SERVICE_YES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]  # Интернет считается отдельно.


class ChurnPreprocessor(TransformerMixin, BaseEstimator):
    """Медиана и LabelEncoder обучаются только внутри текущего train/CV-фолда."""

    def fit(self, X, y=None):
        data = pd.DataFrame(X).drop(columns=["customerID", "Churn"], errors="ignore").copy()  # Исключаем ID и ответ.
        self.input_cols_ = list(data.columns)  # Сохраняем 19 исходных полей.
        self.numeric_medians_ = {c: float(pd.to_numeric(data[c], errors="coerce").median()) for c in NUMERIC}  # Не используем validation.
        self.categorical_cols_ = [c for c in data if c not in NUMERIC]  # SeniorCitizen трактуется как категория.
        self.modes_, self.label_encoders_ = {}, {}  # Сохраняем кодировщики и резервные значения.
        for column in self.categorical_cols_:
            series = data[column].astype("string")  # Числовой бинарный код также приводится к категории.
            self.modes_[column] = str(series.mode().iloc[0])  # На случай настоящего пропуска.
            self.label_encoders_[column] = LabelEncoder().fit(series.fillna(self.modes_[column]).astype(str))  # Фиксируем соответствие.
        self.feature_cols_ = self.input_cols_ + ["TenureYears", "AvgMonthlyCharge", "IsNewClient", "HasMultipleServices"]
        self.fit_rows_ = len(data)  # Метаданные для аудита границы обучения.
        self.n_features_in_ = len(self.input_cols_)  # Стандартное поле sklearn.
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_cols_")  # Заполнение и кодирование должны быть обучены.
        data = pd.DataFrame(X).reindex(columns=self.input_cols_).copy()  # Порядок исходных полей фиксирован.
        for column in NUMERIC:
            values = pd.to_numeric(data[column], errors="coerce")  # Пробелы в TotalCharges становятся NaN.
            data[column] = values.fillna(self.numeric_medians_[column])  # Используем медианы fit, не validation.
        service_count = sum((data[c] == "Yes").astype(int) for c in SERVICE_YES)  # Считаем реально подключённые услуги.
        service_count += data["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)  # Интернет — отдельная услуга.
        data["TenureYears"] = data["tenure"] / 12  # Срок обслуживания в годах.
        data["AvgMonthlyCharge"] = data["TotalCharges"] / (data["tenure"] + 1)  # Защищаем деление при tenure=0.
        data["IsNewClient"] = (data["tenure"] <= 6).astype(int)  # Первые шесть месяцев обслуживания.
        data["HasMultipleServices"] = service_count  # Несмотря на имя, это счётчик, а не бинарный индикатор.
        for column in self.categorical_cols_:
            values = data[column].astype("string").fillna(self.modes_[column]).astype(str)  # Не меняем значения на test.
            encoder = self.label_encoders_[column]  # Используем тот же словарь классов.
            lookup = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))  # Явная схема для неизвестных категорий.
            data[column] = values.map(lookup).fillna(-1).astype(int)  # Новый уровень получает отдельный код -1.
        encoded = data.loc[:, self.feature_cols_]  # Единственный порядок для всех моделей.
        if not np.isfinite(encoded.to_numpy(dtype=float)).all():
            raise ValueError("После обработки остались неконечные значения")
        return encoded

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_cols_")
        return np.array(self.feature_cols_, dtype=object)


def predict_customer(customer, model, preprocessor, feature_cols):
    """Возвращаем класс и модельную вероятность оттока для сырого клиента."""
    if "Churn" in customer:
        raise ValueError("Удалите Churn из входных признаков")  # Ответ не должен попадать в профиль.
    missing = set(preprocessor.input_cols_) - set(customer)
    if missing:
        raise ValueError(f"Нужен полный профиль; отсутствуют поля: {sorted(missing)}")
    raw = pd.DataFrame([dict(customer)])  # Исходный словарь не изменяется.
    for column in ["tenure", "MonthlyCharges"]:
        try:
            number = float(raw.iloc[0][column])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} должен быть числом") from exc
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"{column} должен быть неотрицательным конечным числом")
    encoded = preprocessor.transform(raw)  # Все преобразования согласованы с обучением.
    if list(encoded.columns) != list(feature_cols):
        raise ValueError("Схема признаков не соответствует сохранённой модели")
    probability = float(model.predict_proba(encoded)[0, 1])  # Сбалансированные веса не гарантируют калибровку.
    return int(probability >= .5), probability  # Порог заранее фиксирован.


def load_local_artifacts(models_dir="models"):
    folder = Path(models_dir)
    names = ["rf_model", "xgb_model", "lgbm_model", "cat_model", "tuned_lgbm", "best_model", "preprocessor", "scaler", "label_encoders"]
    objects = {name: joblib.load(folder / f"{name}.pkl") for name in names}  # Ни одна операция не обучает модель.
    objects["feature_cols"] = json.loads((folder / "feature_cols.json").read_text(encoding="utf-8"))
    return objects


def load_artifacts_from_github(repository_url, branch="main"):
    match = re.fullmatch(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?", repository_url.removesuffix(".git"))
    if not match:
        raise ValueError("Ожидается адрес https://github.com/owner/repo")
    base = f"https://raw.githubusercontent.com/{match[1]}/{match[2]}/{branch}/module-3-churn"
    loaded = {}
    for name in ["best_model", "preprocessor"]:
        response = requests.get(f"{base}/models/{name}.pkl", timeout=30)  # Только свой доверенный репозиторий.
        response.raise_for_status()  # Ошибку 404 не маскируем десериализацией.
        loaded[name] = joblib.load(BytesIO(response.content))
    response = requests.get(f"{base}/models/feature_cols.json", timeout=30)
    response.raise_for_status()
    loaded["feature_cols"] = response.json()
    return loaded
