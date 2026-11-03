"""Единая предобработка House Prices, сохранение схемы и инференс."""

from io import BytesIO  # Десериализуем скачанные модели без временных файлов.
from pathlib import Path  # Работаем с переносимыми путями внутри модуля.
import json  # Читаем схему признаков и метаданные.
import re  # Проверяем формат адреса GitHub.

import joblib  # Восстанавливаем обученные объекты.
import numpy as np  # Проверяем конечность чисел и возвращаем цену из log-пространства.
import pandas as pd  # Обрабатываем смешанные числовые и категориальные данные.
import requests  # Загружаем собственные артефакты через GitHub raw.
from sklearn.base import BaseEstimator, TransformerMixin  # Делаем трансформер совместимым с Pipeline и clone.
from sklearn.utils.validation import check_is_fitted  # Запрещаем transform до fit.

QUALITY = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {name: QUALITY for name in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]}  # Сохраняем порядок качества.
ORDINAL_MAPS.update({
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
})  # Для остальных категорий искусственный порядок не задаём.
ABSENCE_CATEGORIES = ["Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu", "MasVnrType",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]  # NA здесь часто означает отсутствие объекта.
ZERO_FEATURES = ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]  # Отсутствующие площади и количества.


class HousePreprocessor(TransformerMixin, BaseEstimator):
    """Оцениваем медианы, категории и схему только внутри fit текущего фолда."""

    def __init__(self, skew_threshold=0.75):
        self.skew_threshold = skew_threshold  # sklearn.clone требует сохранять параметр без изменения.

    def _raw_frame(self, frame):
        data = pd.DataFrame(frame).copy(deep=True)  # Не меняем исходный объект.
        if hasattr(self, "input_cols_"):
            data = data.reindex(columns=self.input_cols_)  # Недостающие поля станут NaN для обученного заполнения.
        else:
            data = data.drop(columns=["Id", "SalePrice", "SalePrice_log"], errors="ignore")  # Исключаем ID и обе формы цели.
        return data  # Pandas распознаёт None и NaN как пропуски без неявного изменения dtype.

    def _fill(self, frame):
        data = self._raw_frame(frame)  # Выбираем ровно обученную входную схему.
        for name in self.numeric_input_cols_:
            data[name] = pd.to_numeric(data[name], errors="raise")  # Ошибочный текст не превращается в медиану.
        for name in ABSENCE_CATEGORIES:
            if name in data:
                data[name] = data[name].fillna("None")  # Кодируем смысловое отсутствие отдельным уровнем.
        for name in ZERO_FEATURES:
            if name in data:
                data[name] = data[name].fillna(0)  # Суммы площадей корректно учитывают отсутствие строения.
        if "LotFrontage" in data:
            group_fill = data["Neighborhood"].map(self.frontage_medians_)  # Районные медианы только из fit.
            data["LotFrontage"] = data["LotFrontage"].fillna(group_fill)  # Восстанавливаем типичную длину фасада района.
        for name in self.numeric_input_cols_:
            data[name] = data[name].fillna(self.numeric_medians_[name])  # Общая медиана служит резервом.
        for name in self.category_input_cols_:
            data[name] = data[name].fillna(self.category_modes_[name]).astype(str)  # Остальные категории заполняем модой.
        return data  # На этом этапе все поля имеют значение.

    @staticmethod
    def _engineer(data):
        result = data.copy()  # Сохраняем заполненные исходные признаки.
        result["TotalSF"] = result["TotalBsmtSF"] + result["1stFlrSF"] + result["2ndFlrSF"]  # Общая площадь уровней.
        result["TotalBath"] = result["FullBath"] + .5 * result["HalfBath"] + result["BsmtFullBath"] + .5 * result["BsmtHalfBath"]
        porch_cols = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]  # Все открытые и закрытые площадки.
        result["TotalPorch"] = result[porch_cols].sum(axis=1)  # Суммарная площадь крылец и настилов.
        result["HouseAge"] = result["YrSold"] - result["YearBuilt"]  # Возраст на дату продажи, не на текущую дату.
        result["RemodAge"] = result["YrSold"] - result["YearRemodAdd"]  # Время после последней реконструкции.
        result["HasPool"] = (result["PoolArea"] > 0).astype(int)  # Бинарный индикатор бассейна.
        result["HasGarage"] = (result["GarageArea"] > 0).astype(int)  # Бинарный индикатор гаража.
        result["Has2ndFloor"] = (result["2ndFlrSF"] > 0).astype(int)  # Наличие второго этажа.
        result["HasFireplace"] = (result["Fireplaces"] > 0).astype(int)  # Наличие камина.
        result["OverallQual_TotalSF"] = result["OverallQual"] * result["TotalSF"]  # Взаимодействие площади и качества.
        result["WasRemodeled"] = (result["YearRemodAdd"] > result["YearBuilt"]).astype(int)  # Факт реконструкции.
        return result  # Целевую переменную не используем.

    def _encode(self, engineered):
        encoded = engineered.copy()  # Кодируем признаки на независимой копии.
        for name, mapping in ORDINAL_MAPS.items():
            unknown = ~encoded[name].isin(mapping)  # Проверяем новые или ошибочные уровни качества.
            encoded[name] = encoded[name].map(mapping)  # Применяем явный, а не алфавитный порядок.
            encoded.loc[unknown, name] = self.ordinal_defaults_[name]  # Незнакомое качество заменяем обученной модой.
        for name in self.nominal_cols_:
            encoded[name] = pd.Categorical(encoded[name].astype(str), categories=self.categories_[name])  # Фиксируем категории train.
        encoded = pd.get_dummies(encoded, columns=self.nominal_cols_, drop_first=True, dtype=float)  # Базовый уровень исключается.
        for name in self.log_cols_:
            encoded[name] = np.log1p(encoded[name].clip(lower=0))  # Скошенные неотрицательные признаки смягчаем log1p.
        return encoded.astype(float)  # StandardScaler получает только числовые значения.

    def fit(self, X, y=None):
        raw = self._raw_frame(X)  # y не участвует в обучении преобразований.
        self.input_cols_ = list(raw.columns)  # Сохраняем исходные 79 полей и порядок.
        self.numeric_input_cols_ = raw.select_dtypes(include="number").columns.tolist()  # Типы определяем только по fit.
        self.category_input_cols_ = [c for c in raw if c not in self.numeric_input_cols_]  # Остальные поля категориальные.
        self.numeric_medians_ = {c: float(raw[c].median()) if raw[c].notna().any() else 0.0 for c in self.numeric_input_cols_}
        self.frontage_medians_ = raw.groupby("Neighborhood")["LotFrontage"].median().to_dict()  # Заполнение по районам.
        self.category_modes_ = {c: (str(raw[c].mode().iloc[0]) if raw[c].notna().any() else "None") for c in self.category_input_cols_}
        filled = self._fill(raw)  # Используем только что обученные параметры заполнения.
        self.ordinal_defaults_ = {c: ORDINAL_MAPS[c].get(str(filled[c].mode().iloc[0]), 0) for c in ORDINAL_MAPS}
        engineered = self._engineer(filled)  # Создаём все новые признаки до выбора кодирования.
        for name in ["MSSubClass", "MoSold"]:
            engineered[name] = engineered[name].astype(int).astype(str)  # Тип здания и месяц являются категориями.
        self.nominal_cols_ = [c for c in engineered if c not in ORDINAL_MAPS and not pd.api.types.is_numeric_dtype(engineered[c])]
        self.categories_ = {c: sorted(engineered[c].astype(str).unique()) for c in self.nominal_cols_}  # Test не расширяет схему.
        self.log_cols_ = [c for c in engineered.select_dtypes(include="number")
                          if engineered[c].min() >= 0 and engineered[c].nunique() > 10
                          and engineered[c].skew() > self.skew_threshold]  # Выбираем преобразования только по train.
        encoded = self._encode(engineered)  # Получаем полный числовой набор.
        self.feature_cols_ = [c for c in encoded if encoded[c].nunique() > 1]  # Убираем постоянные признаки обучения.
        self.fit_rows_ = len(raw)  # Сохраняем размер обучающей части для аудита.
        self.n_features_in_ = len(self.input_cols_)  # Метаданные sklearn.
        return self  # Совместимость с Pipeline/GridSearchCV.

    def transform(self, X):
        check_is_fitted(self, "feature_cols_")  # Предотвращаем использование необученного заполнения.
        engineered = self._engineer(self._fill(X))  # Применяем сохранённые статистики.
        for name in ["MSSubClass", "MoSold"]:
            engineered[name] = engineered[name].astype(int).astype(str)  # Повторяем то же кодирование.
        encoded = self._encode(engineered).reindex(columns=self.feature_cols_, fill_value=0)  # Восстанавливаем порядок.
        if not np.isfinite(encoded.to_numpy()).all():
            raise ValueError("После обработки обнаружены NaN или бесконечные значения")
        return encoded  # Никакого повторного fit.

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_cols_")  # Метод нужен для интерпретации модели.
        return np.array(self.feature_cols_, dtype=object)  # Возвращаем фактическую схему.


def validate_house(house):
    """Проверяем минимальные поля пользовательского профиля."""
    data = dict(house)  # Оригинальный словарь остаётся неизменным.
    for name in ["OverallQual", "GrLivArea", "YearBuilt"]:
        if name not in data or pd.isna(data[name]):
            raise ValueError(f"Обязательное поле: {name}")
    for name in ["OverallQual", "GrLivArea", "YearBuilt", "YrSold", "GarageCars"]:
        if name in data and pd.notna(data[name]):
            try:
                value = float(data[name])  # Строковые числа допускаются после явной проверки.
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{name} должен быть числом") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} должен быть конечным неотрицательным числом")
            data[name] = value
    if not 1 <= data["OverallQual"] <= 10 or data["OverallQual"] % 1:
        raise ValueError("OverallQual должен быть целым от 1 до 10")
    if data["GrLivArea"] <= 0:
        raise ValueError("GrLivArea должен быть положительным")
    if "YrSold" in data and pd.notna(data["YrSold"]) and data["YearBuilt"] > data["YrSold"]:
        raise ValueError("Год постройки не может быть позже года продажи")
    return data


def predict_price(house_data, model, scaler, feature_cols, preprocessor):
    """Сырой профиль -> сохранённые преобразования -> прогноз в исторических USD."""
    house = validate_house(house_data)  # Проверяем ввод до заполнения остальных полей.
    forbidden = {"SalePrice", "SalePrice_log"} & set(house)  # Ответы не должны попадать во вход.
    if forbidden:
        raise ValueError("Удалите целевую цену из входного профиля")
    encoded = preprocessor.transform(pd.DataFrame([house]))  # Единая логика с обучением и CV.
    if list(encoded.columns) != list(feature_cols):
        raise ValueError("Список признаков не соответствует сохранённому preprocessor")
    log_price = float(model.predict(scaler.transform(encoded))[0])  # Модели обучены на log1p(SalePrice).
    price = float(np.expm1(log_price))  # Возвращаем исходные единицы.
    if not np.isfinite(price) or price <= 0:
        raise ValueError("Модель выдала некорректную цену; проверьте выход за диапазон данных")
    return price


def interactive_predict(bundle):
    """Необязательный ввод; остальные поля заполняются статистиками обучения."""
    house = {"OverallQual": int(input("Качество, 1-10: ")),
             "GrLivArea": float(input("Жилая площадь, кв. футы: ")),
             "YearBuilt": int(input("Год постройки: ")),
             "YrSold": int(input("Год продажи: ")),
             "GarageCars": int(input("Вместимость гаража: "))}  # Это сокращённый профиль.
    price = predict_price(house, bundle["best_model"], bundle["scaler"], bundle["feature_cols"], bundle["preprocessor"])
    print(f"Оценка по историческим данным Ames: ${price:,.0f}")
    return price


def load_local_artifacts(models_dir="models"):
    folder = Path(models_dir)  # Пользователь запускает функцию из папки модуля.
    names = ["lr_model", "ridge_model", "lasso_model", "enet_model", "gbr_model", "best_model", "scaler", "preprocessor"]
    loaded = {name: joblib.load(folder / f"{name}.pkl") for name in names}  # Согласованный комплект, без fit.
    loaded["feature_cols"] = json.loads((folder / "feature_cols.json").read_text(encoding="utf-8"))
    return loaded


def load_artifacts_from_github(repository_url, branch="main"):
    match = re.fullmatch(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?", repository_url.removesuffix(".git"))
    if not match:
        raise ValueError("Ожидается адрес корня репозитория https://github.com/owner/repo")
    base = f"https://raw.githubusercontent.com/{match[1]}/{match[2]}/{branch}/module-2-house-prices"
    loaded = {}  # Загружаем только собственные доверенные joblib-файлы.
    for name in ["best_model", "scaler", "preprocessor"]:
        response = requests.get(f"{base}/models/{name}.pkl", timeout=30)
        response.raise_for_status()  # 404 не должен маскироваться под ошибку десериализации.
        loaded[name] = joblib.load(BytesIO(response.content))
    response = requests.get(f"{base}/models/feature_cols.json", timeout=30)
    response.raise_for_status()
    loaded["feature_cols"] = response.json()  # Используем схему того же репозитория.
    loaded["base_url"] = base
    return loaded
