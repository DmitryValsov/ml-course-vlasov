"""Предобработка и инференс для практической работы по Titanic."""

from io import BytesIO  # Читаем скачанные модели без временных файлов.
from pathlib import Path  # Используем переносимые пути к артефактам.
import json  # Читаем список признаков и конфигурацию.
import re  # Проверяем структуру адреса GitHub.

import joblib  # Сохраняем и восстанавливаем обученные объекты.
import numpy as np  # Проверяем конечность числовых значений.
import pandas as pd  # Преобразуем словари пассажиров в таблицы.
import requests  # Загружаем артефакты по raw-ссылкам.
from sklearn.preprocessing import LabelEncoder  # Фиксируем кодировку пола.

FEATURE_COLS = [  # Порядок столбцов одинаков при обучении и инференсе.
    "Pclass", "Sex_Encoded", "Age", "SibSp", "Parch", "Fare",
    "Family_Size", "Is_Alone", "Embarked_Q", "Embarked_S",
]
REQUIRED_FIELDS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def fit_preprocessor(frame):
    """Оцениваем параметры заполнения только по обучающей части."""
    encoder = LabelEncoder().fit(frame["Sex"])  # Не задаём обратную ручную кодировку.
    grouped = frame.groupby(["Pclass", "Sex"])["Age"].median()  # Учитываем класс и пол.
    return {  # Словарь можно сохранить без зависимости от пользовательского класса.
        "age_medians": grouped.to_dict(),  # Используем эти медианы и для новых пассажиров.
        "age_global": float(frame["Age"].median()),  # Резерв для отсутствующей группы.
        "fare_median": float(frame["Fare"].median()),  # Заполняем пропуски стоимости.
        "embarked_mode": str(frame["Embarked"].mode().iloc[0]),  # Заполняем порт модой.
        "le_sex": encoder,  # Сохраняем обученный LabelEncoder.
        "fit_rows": len(frame),  # Позволяет проверить отсутствие утечки.
    }


def validate_passengers(frame):
    """Проверяем обязательные поля, диапазоны и допустимые категории."""
    absent = set(REQUIRED_FIELDS) - set(frame.columns)  # Находим отсутствующие столбцы.
    if absent:  # Ошибка должна явно называть недостающие поля.
        raise ValueError(f"Отсутствуют поля: {sorted(absent)}")
    result = frame.copy(deep=True)  # Не меняем переданный пользователем объект.
    for name in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:  # Проверяем числа отдельно.
        try:  # Некорректная строка не должна превращаться в пропуск.
            result[name] = pd.to_numeric(result[name], errors="raise")
        except (ValueError, TypeError) as exc:  # Возвращаем понятное имя поля.
            raise ValueError(f"Поле {name} должно быть числом") from exc
        known = result[name].dropna()  # Пропуски Age и Fare допускаются до заполнения.
        if not np.isfinite(known).all():  # Бесконечности нельзя подавать модели.
            raise ValueError(f"Поле {name} должно содержать конечные числа")
    for name in ["Pclass", "SibSp", "Parch"]:  # Эти поля должны быть целыми и известными.
        if result[name].isna().any() or (result[name] % 1 != 0).any():
            raise ValueError(f"Поле {name} должно быть целым и не содержать пропусков")
    if not result["Pclass"].isin([1, 2, 3]).all():  # Классы билета ограничены данными.
        raise ValueError("Pclass должен быть 1, 2 или 3")
    if (result[["Age", "SibSp", "Parch", "Fare"]] < 0).any().any():  # Отсекаем отрицательные числа.
        raise ValueError("Age, SibSp, Parch и Fare должны быть неотрицательными")
    if not result["Sex"].isin(["female", "male"]).all():  # Не кодируем неизвестный пол молча.
        raise ValueError("Sex должен быть female или male")
    known_ports = result["Embarked"].dropna()  # Пропущенный порт заполнится сохранённой модой.
    if not known_ports.isin(["C", "Q", "S"]).all():  # Неизвестный порт не равен базовой категории.
        raise ValueError("Embarked должен быть C, Q или S")
    return result  # Возвращаем независимую проверенную копию.


def transform_passengers(frame, state, feature_cols=None):
    """Заполняем, создаём и кодируем признаки без повторного fit."""
    data = validate_passengers(frame)  # Единая проверка для train, validation и инференса.
    medians = [state["age_medians"].get((p, s), state["age_global"])
               for p, s in zip(data["Pclass"], data["Sex"])]  # Медианы берём только из train.
    data["Age"] = data["Age"].fillna(pd.Series(medians, index=data.index))  # Сохраняем индексы.
    data["Fare"] = data["Fare"].fillna(state["fare_median"])  # Используем общую медиану train.
    data["Embarked"] = data["Embarked"].fillna(state["embarked_mode"])  # Сохраняем исходную категорию.
    data = data.drop(columns=["Cabin"], errors="ignore")  # Cabin исключён из-за доли пропусков.
    data["Family_Size"] = data["SibSp"] + data["Parch"] + 1  # Включаем самого пассажира.
    data["Is_Alone"] = (data["Family_Size"] == 1).astype(int)  # Отдельный признак одиночного путешествия.
    data["Age_Group"] = pd.cut(data["Age"], bins=[-np.inf, 12, 18, 35, 60, np.inf],
                               labels=["0-12", "13-18", "19-35", "36-60", "61+"])  # Фиксированные интервалы.
    data["Sex_Encoded"] = state["le_sex"].transform(data["Sex"])  # female=0, male=1 для этих данных.
    ports = pd.Categorical(data["Embarked"], categories=["C", "Q", "S"])  # Полная схема категорий.
    encoded = pd.get_dummies(ports, prefix="Embarked", drop_first=True, dtype=int)  # C - базовый порт.
    encoded.index = data.index  # Восстанавливаем индексы после создания dummy-столбцов.
    data = pd.concat([data, encoded], axis=1)  # Оригинальный Embarked остаётся для анализа.
    columns = list(feature_cols) if feature_cols is not None else FEATURE_COLS  # Фиксируем порядок.
    features = data.loc[:, columns].astype(float)  # Все модели получают числовую матрицу.
    if not np.isfinite(features.to_numpy()).all():  # Пропуски после обработки недопустимы.
        raise ValueError("После предобработки остались нечисловые значения или пропуски")
    return data, features  # Первая таблица для EDA, вторая для модели.


def predict_passenger(passenger_data, model, scaler, feature_cols,
                      use_scaling=True, preprocessor=None):
    """Возвращаем (класс, вероятность выживания), не изменяя входной словарь."""
    if preprocessor is None:  # Параметры заполнения обязательны для воспроизводимого инференса.
        raise ValueError("Передайте сохранённый preprocessor")
    raw = pd.DataFrame([dict(passenger_data)])  # Создаём независимую строку пассажира.
    _, features = transform_passengers(raw, preprocessor, feature_cols)  # Повторяем обучение признаков.
    if use_scaling:  # Масштабирование включаем только для линейной модели.
        if scaler is None:  # Не используем случайный или необученный scaler.
            raise ValueError("Для Logistic Regression нужен обученный scaler")
        features = scaler.transform(features)  # Только transform, никогда fit_transform.
    prediction = int(model.predict(features)[0])  # Получаем итоговый бинарный класс.
    probability = float(model.predict_proba(features)[0, 1])  # Вероятность класса Survived=1.
    return prediction, probability  # Совместимый результат для обеих обязательных моделей.


def interactive_predict(model, scaler, feature_cols, preprocessor):
    """Необязательный ввод с клавиатуры; не запускается при Run All."""
    passenger = {  # Собираем значения в тех же единицах, что и в CSV.
        "Pclass": int(input("Класс билета (1/2/3): ")),
        "Sex": input("Пол (female/male): ").strip().lower(),
        "Age": float(input("Возраст, лет: ")),
        "SibSp": int(input("Братья, сёстры, супруги (SibSp): ")),
        "Parch": int(input("Родители и дети (Parch): ")),
        "Fare": float(input("Стоимость билета (Fare): ")),
        "Embarked": input("Порт посадки (C/Q/S): ").strip().upper(),
    }
    result = predict_passenger(passenger, model, scaler, feature_cols,
                               preprocessor=preprocessor)  # Применяем общую функцию.
    print(f"Предсказанный класс: {result[0]}; вероятность выживания: {result[1]:.1%}")
    return result  # Позволяет проверить интерактивный режим программно.


def load_local_artifacts(models_dir="models"):
    """Загружаем локальные артефакты без обучения."""
    folder = Path(models_dir)  # Все пути отсчитываются от папки модуля.
    loaded = {key: joblib.load(folder / f"{key}.pkl") for key in
              ["lr_model", "dt_model", "rf_model", "scaler", "preprocessor"]}  # Согласованный комплект.
    loaded["feature_cols"] = json.loads((folder / "feature_cols.json").read_text(encoding="utf-8"))
    return loaded  # Encoder находится и отдельно, и в состоянии предобработки.


def load_artifacts_from_github(repository_url, branch="main"):
    """Загружаем доверенный комплект через настоящие raw-ссылки GitHub."""
    match = re.fullmatch(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?",
                         repository_url.removesuffix(".git"))  # Не принимаем произвольный сервер.
    if not match:  # Адрес должен указывать на корень репозитория.
        raise ValueError("Ожидается адрес https://github.com/owner/ml-course-vlasov")
    owner, repo = match.groups()  # Извлекаем владельца и реальное имя репозитория.
    base_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/module-1-titanic"
    loaded = {}  # Собираем модели и все параметры преобразования.
    for key in ["lr_model", "dt_model", "rf_model", "scaler", "preprocessor"]:
        response = requests.get(f"{base_url}/models/{key}.pkl", timeout=30)  # Ограничиваем ожидание.
        response.raise_for_status()  # HTML-ошибка 404 не должна передаваться в joblib.
        loaded[key] = joblib.load(BytesIO(response.content))  # Загружаем только свой доверенный .pkl.
    response = requests.get(f"{base_url}/models/feature_cols.json", timeout=30)  # Получаем схему признаков.
    response.raise_for_status()  # Ошибку загрузки показываем явно.
    loaded["feature_cols"] = response.json()  # Используем сохранённый порядок, а не ручной список.
    loaded["base_url"] = base_url  # Адрес выводится в проверке раздела 11.
    return loaded  # Ни одна из операций не выполняет fit.
