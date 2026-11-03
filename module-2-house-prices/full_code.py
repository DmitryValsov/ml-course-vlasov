# Зависимости requirements.txt:
# numpy==2.2.6
# pandas==2.2.3
# scipy==1.18.1
# matplotlib==3.10.1
# seaborn==0.13.2
# scikit-learn==1.7.2
# joblib==1.5.2
# requests==2.32.5
# tabulate==0.9.0
# nbformat==5.10.4
# nbclient==0.10.2
# nbconvert==7.16.6
# ipykernel==6.30.1

import os  # Настраиваем относительные пути.
# %% Настройка самостоятельного запуска
from pathlib import Path  # Находим папку модуля.

import matplotlib  # Скрипт сохраняет графики без открытия окон.

matplotlib.use("Agg")  # Для автоматического выполнения.
os.chdir(Path(__file__).resolve().parent)  # Рабочая папка всегда соответствует модулю.

# %% Общие функции: полный исходный код доступен ниже в приложении
from house_utils import HousePreprocessor, predict_price, interactive_predict
from house_utils import load_local_artifacts, load_artifacts_from_github

# %% Кодовая ячейка 5
from pathlib import Path  # Используем относительные пути внутри модуля.
from time import perf_counter  # Разделяем время подбора и финального fit.
import json, platform, warnings  # Сохраняем происхождение, версии и предупреждения.
import numpy as np  # Логарифмируем цену и проверяем численные результаты.
import pandas as pd  # Загружаем таблицы и строим описания данных.
import matplotlib.pyplot as plt  # Строим графики и сохраняем PNG.
from matplotlib.ticker import FuncFormatter  # Сокращаем длинные денежные подписи осей.
import seaborn as sns  # Статистические диаграммы EDA.
import sklearn, joblib  # Фиксируем версии и сохраняем обученные объекты.
from IPython.display import display, Markdown  # Выводим таблицы в ноутбук.
from sklearn.base import clone  # Создаём независимые модели при проверке.
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score  # Holdout и CV.
from sklearn.pipeline import Pipeline  # Предобработка обучается внутри каждого CV-фолда.
from sklearn.preprocessing import StandardScaler  # Масштаб нужен для регуляризации.
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  # Четыре обязательные модели.
from sklearn.ensemble import GradientBoostingRegressor  # Бонусная нелинейная модель.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Метрики регрессии.

sns.set_theme(style="whitegrid", font="DejaVu Sans")  # Русские подписи отображаются без замены символов.
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 165})  # Читаемые изображения для отчёта.
config = json.loads(Path("student.json").read_text(encoding="utf-8"))  # Сведения об авторе и GitHub.
train_df = pd.read_csv("data/train.csv")  # Исходные 1460 размеченных домов.
test_df = pd.read_csv("data/test.csv")  # 1459 домов без известной цены.
print(train_df.shape, test_df.shape)  # Обязательная проверка размеров.
display(train_df.head())  # Первичный осмотр записей.
train_df.info()  # Типы и заполненность всех столбцов.
display(train_df.describe())  # Числовые распределения и диапазоны.
print("Дубликаты Id:", train_df["Id"].duplicated().sum(), "Полные дубликаты:", train_df.duplicated().sum())  # Проверяем повторения.
assert train_df.shape == (1460, 81) and test_df.shape == (1459, 80)  # Проверяем нужный вариант набора.
assert "SalePrice" not in test_df and train_df["Id"].tolist() == list(range(1, 1461))  # Защита от подмены данных.
assert test_df["Id"].tolist() == list(range(1461, 2920))  # Проверяем диапазон Kaggle test.
raw_X = train_df.drop(columns=["Id", "SalePrice"])  # ID и целевая никогда не являются признаками.
missing_count = train_df.isnull().sum()  # Находим пропуски по исходным столбцам.
schema = pd.DataFrame({"dtype": train_df.dtypes.astype(str), "missing_count": missing_count,
                       "missing_pct": missing_count / len(train_df) * 100})  # Полная схема всех 81 столбца.
display(schema.loc[schema["missing_count"] > 0].sort_values("missing_count", ascending=False))  # Смотрим 19 проблемных полей.
price_stats = {"mean": float(train_df["SalePrice"].mean()), "median": float(train_df["SalePrice"].median()),
               "skew": float(train_df["SalePrice"].skew()), "kurtosis": float(train_df["SalePrice"].kurtosis()),
               "log_skew": float(np.log1p(train_df["SalePrice"]).skew())}  # Получаем фактические, не примерные статистики.
print("Статистики SalePrice:", price_stats)  # Проверяем асимметрию до и после log1p.
print("Числовые / категориальные признаки по dtype:", len(raw_X.select_dtypes("number").columns), len(raw_X.select_dtypes("object").columns))  # Не смешиваем тип хранения и смысл.
info = "# House Prices: описание набора\n\nИсточник: Kaggle; учебная копия D2L, URLs и SHA-256 в sources.json.\n\n"  # Указываем происхождение.
info += "Train: 1460×81, test: 1459×80. 79 входных признаков, Id и SalePrice. Площади в sq ft, цены в USD.\n\n"  # Схема набора.
info += schema.round(3).to_markdown() + "\n\nСтатистики цели:\n" + json.dumps(price_stats, ensure_ascii=False, indent=2)  # Полная схема и статистики.
Path("data/house_prices_info.md").write_text(info, encoding="utf-8")  # Обязательное описание данных.

# %% Кодовая ячейка 8
fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))  # Шесть обязательных панелей EDA.
sns.histplot(train_df["SalePrice"], bins=35, kde=True, color="#4b83b4", ax=axes[0, 0])  # Исходная цена.
axes[0, 0].set(title="SalePrice до преобразования", xlabel="Цена, USD", ylabel="Число домов")  # Подписываем единицы.
axes[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value / 1000:.0f}"))  # Подписи не сливаются.
axes[0, 0].set_xlabel("Цена, тыс. USD")  # Единицы соответствуют сокращённым значениям.
sns.histplot(np.log1p(train_df["SalePrice"]), bins=35, kde=True, color="#4d9a83", ax=axes[0, 1])  # Та же цель в логарифмах.
axes[0, 1].set(title="SalePrice после log1p", xlabel="log(1 + SalePrice)", ylabel="Число домов")  # Не смешиваем шкалы.
sns.boxplot(data=train_df, x="OverallQual", y="GrLivArea", color="#87a7c9", ax=axes[0, 2])  # Площадь внутри уровней качества.
axes[0, 2].set(title="Площадь по общему качеству", xlabel="OverallQual, балл", ylabel="GrLivArea, кв. футы")  # Указываем смысл полей.
corr_cols = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "YearBuilt"]  # Выбираем ключевые числовые признаки.
sns.heatmap(train_df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1, 0])  # Проверяем корреляции.
axes[1, 0].set(title="Корреляции ключевых столбцов", xlabel="Признак", ylabel="Признак")  # Явно отмечаем обе оси.
special = (train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000)  # Предзаданное предметное правило.
axes[1, 1].scatter(train_df["GrLivArea"], train_df["SalePrice"], s=15, alpha=.45, label="Все дома")  # Все исходные точки.
axes[1, 1].scatter(train_df.loc[special, "GrLivArea"], train_df.loc[special, "SalePrice"], color="#bf5b59", s=50, label="Предметное правило")  # Выделяем две аномалии.
axes[1, 1].set(title="Площадь и цена: необычные дома", xlabel="GrLivArea, кв. футы", ylabel="Цена, USD")  # Подписываем единицы.
axes[1, 1].legend(fontsize=9)  # Необычные точки видны отдельно.
missing_pct = (missing_count[missing_count > 0] / len(train_df) * 100).sort_values()  # Считаем доли всех пропусков.
missing_pct.plot.barh(ax=axes[1, 2], color="#cc9558")  # Показываем все 19 столбцов с NA.
axes[1, 2].set(title="Пропуски в исходном train", xlabel="Пропуски, %", ylabel="Столбец", xlim=(0, 100))  # Единая шкала процентов.
axes[1, 2].tick_params(axis="y", labelsize=8)  # Имена всех столбцов остаются читаемыми.
fig.suptitle("House Prices: разведочный анализ", fontsize=17)  # Общий заголовок исследования.
fig.tight_layout(rect=[0, 0, 1, .96])  # Предотвращаем обрезку подписей.
fig.savefig("examples/eda_plots.png", bbox_inches="tight")  # Сохраняем обязательный рисунок.
plt.show()  # Выводим EDA в выполненный ноутбук.
display(train_df.loc[special, ["Id", "GrLivArea", "SalePrice"]])  # Проверяем идентификаторы двух необычных домов.

# %% Кодовая ячейка 10
outer_train_ids, valid_ids = train_test_split(train_df["Id"].to_numpy(), test_size=.2, random_state=42)  # Сначала откладываем сырые строки.
by_id = train_df.set_index("Id", drop=False)  # Сохраняем порядок объектов split.
outer_train = by_id.loc[outer_train_ids].copy()  # Только эта часть доступна для подготовки модели.
valid_frame = by_id.loc[valid_ids].copy()  # 292 дома не участвуют в CV или заполнении пропусков.
q1, q3 = outer_train["GrLivArea"].quantile([.25, .75])  # Оцениваем IQR только по обучающей части.
iqr = q3 - q1  # Межквартильный размах.
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr  # Диагностические границы площади.
iqr_flag = ~outer_train["GrLivArea"].between(lower, upper)  # Крупные дома остаются, если нет предметной причины удаления.
domain_flag = (outer_train["GrLivArea"] > 4000) & (outer_train["SalePrice"] < 300000)  # Применяем заранее заданное правило.
removed_ids = outer_train.loc[domain_flag, "Id"].astype(int).tolist()  # Фиксируем удалённые объекты для воспроизведения.
development = outer_train.loc[~domain_flag].copy()  # Обучение и CV используют 1166 домов.
X_train = development.drop(columns=["Id", "SalePrice"])  # Сырые признаки передаются в Pipeline.
y_train_log = np.log1p(development["SalePrice"])  # Цель преобразуется без обучения параметров.
X_valid = valid_frame.drop(columns=["Id", "SalePrice"])  # Исходные поля validation.
y_valid_real = valid_frame["SalePrice"].astype(float)  # Фактические долларовые цены.
y_valid_log = np.log1p(y_valid_real)  # Отдельная шкала для log-метрик.
print("Outer train:", len(outer_train), "После правила:", len(X_train), "Validation:", len(X_valid))  # Проверяем размеры.
print("IQR bounds:", lower, upper, "flags:", int(iqr_flag.sum()), "Удалены Id:", removed_ids)  # Сопоставляем два метода.
preview_preprocessor = HousePreprocessor().fit(X_train)  # Только демонстрация преобразований на обучающей части.
preview_features = preview_preprocessor.transform(X_train)  # Проверяем числа и пропуски.
print("Число подготовленных признаков:", preview_features.shape[1])  # One-Hot расширяет схему.
print("Числовые log1p-признаки:", preview_preprocessor.log_cols_)  # Выбор сделан только по train.
display(preview_features.head())  # Показываем кодирование и новые признаки.
assert np.isfinite(preview_features.to_numpy()).all()  # После обработки NaN недопустимы.
assert not set(development["Id"]).intersection(valid_ids)  # Проверяем разделение данных.

# %% Кодовая ячейка 13
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # Одни и те же фолды для сравнения моделей.
estimators = {"linear_regression": LinearRegression(), "ridge": Ridge(),
              "lasso": Lasso(max_iter=50000, tol=1e-5, random_state=42),
              "elasticnet": ElasticNet(max_iter=50000, tol=1e-5, random_state=42),
              "gradient_boosting": GradientBoostingRegressor(n_estimators=350, learning_rate=.04, max_depth=2,
                                                             min_samples_leaf=5, loss="huber", random_state=42)}  # Фиксированные алгоритмы.
model_names = {"linear_regression": "Linear Regression", "ridge": "Ridge", "lasso": "Lasso",
               "elasticnet": "ElasticNet", "gradient_boosting": "Gradient Boosting"}  # Подписи таблиц и графиков.
grids = {"ridge": {"regressor__alpha": [.1, 1, 10, 30, 100]},
         "lasso": {"regressor__alpha": [.0001, .0003, .0005, .001, .003]},
         "elasticnet": {"regressor__alpha": [.0003, .001, .003], "regressor__l1_ratio": [.2, .5, .8]}}  # Сетки только для train CV.
fitted_pipelines, cv_results, timing, training_warnings = {}, {}, {}, []  # Собираем реальные результаты fit и CV.
for key, estimator in estimators.items():  # Весь подбор проводится до просмотра holdout-метрик.
    pipeline = Pipeline([("preprocessor", HousePreprocessor()), ("scaler", StandardScaler()), ("regressor", estimator)])  # Нет утечки через transform.
    started = perf_counter()  # Полная длительность включает CV и финальный fit.
    with warnings.catch_warnings(record=True) as caught:  # Не теряем сведения о проблемах сходимости.
        warnings.simplefilter("always")  # Предупреждения сохраняются в протоколе.
        if key in grids:  # Подбираем только параметры регуляризации.
            search = GridSearchCV(pipeline, grids[key], cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise", refit=True)
            search.fit(X_train, y_train_log)  # Каждый CV-fit заново обучает медианы, категории и scaler.
            fitted_pipelines[key] = search.best_estimator_  # Уже переобучен на всём development.
            best_index = search.best_index_  # Выбранная строка таблицы CV.
            cv_results[key] = {"rmse_log_mean": float(-search.best_score_),
                "rmse_log_std": float(search.cv_results_["std_test_score"][best_index]),
                "params": search.best_params_}  # Условия выбора модели.
            fit_time = search.refit_time_  # Время финального fit отдельно от полного подбора.
            pd.DataFrame(search.cv_results_).to_csv(f"examples/cv_{key}.csv", index=False)  # Сохраняем все проверенные комбинации.
        else:  # OLS и бонусная модель имеют заранее фиксированные параметры.
            scores = -cross_val_score(pipeline, X_train, y_train_log, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise")
            cv_results[key] = {"rmse_log_mean": float(scores.mean()), "rmse_log_std": float(scores.std()), "params": {}}  # Тот же критерий CV.
            fit_started = perf_counter()  # Отдельный финальный fit на development.
            fitted_pipelines[key] = pipeline.fit(X_train, y_train_log)  # Holdout не используется.
            fit_time = perf_counter() - fit_started  # Измеряем предобработку и обучение вместе.
        training_warnings.extend({"model": key, "category": w.category.__name__, "message": str(w.message)} for w in caught)  # Проверяем сходимость после обучения.
    timing[key] = {"fit_time_sec": float(fit_time), "search_time_sec": float(perf_counter() - started)}  # Два разных вида времени.
    print(model_names[key], "CV RMSE(log):", round(cv_results[key]["rmse_log_mean"], 5), "params:", cv_results[key]["params"])  # Видимый ход подбора.
best_key = min(cv_results, key=lambda k: cv_results[k]["rmse_log_mean"])  # Выбираем модель до оценки holdout.
best_pipeline = fitted_pipelines[best_key]  # Это выбранный алгоритм, а не победитель перебора holdout.
preprocessor = best_pipeline.named_steps["preprocessor"]  # Согласованный трансформер итоговой модели.
scaler = best_pipeline.named_steps["scaler"]  # Сохранённый StandardScaler того же fit.
feature_cols = list(preprocessor.get_feature_names_out())  # Фактический порядок входов регрессора.
best_model = best_pipeline.named_steps["regressor"]  # Отдельный estimator для требуемого predict_price.
display(pd.DataFrame({k: {**cv_results[k], **timing[k]} for k in estimators}).T)  # CV, параметры и время.
print("Выбор по CV:", model_names[best_key], "Число предупреждений:", len(training_warnings))  # Предупреждения не скрываются.
assert not any(w["category"] == "ConvergenceWarning" for w in training_warnings)  # Незавершённое обучение нельзя принять за готовый результат.

# %% Кодовая ячейка 16
metrics, valid_predictions = {}, {}  # Собираем оценки без дополнительного подбора параметров.
for key, pipeline in fitted_pipelines.items():  # Одни и те же 292 дома для всех моделей.
    pred_log = pipeline.predict(X_valid)  # Pipeline применяет только обученные преобразования.
    pred_real = np.expm1(pred_log)  # Переводим логарифмы в исторические доллары.
    assert np.isfinite(pred_real).all()  # Численное переполнение нельзя выдавать за результат.
    train_pred = pipeline.predict(X_train)  # Диагностика разрыва обучения и holdout.
    valid_predictions[key] = {"log": pred_log, "real": pred_real}  # Не пересчитываем предсказания для разных графиков.
    metrics[key] = {"rmse_log": float(np.sqrt(mean_squared_error(y_valid_log, pred_log))),
                    "mae_log": float(mean_absolute_error(y_valid_log, pred_log)), "r2_log": float(r2_score(y_valid_log, pred_log)),
                    "rmse_real": float(np.sqrt(mean_squared_error(y_valid_real, pred_real))),
                    "mae_real": float(mean_absolute_error(y_valid_real, pred_real)), "r2_real": float(r2_score(y_valid_real, pred_real)),
                    "train_rmse_log": float(np.sqrt(mean_squared_error(y_train_log, train_pred))),
                    "cv_rmse_log": cv_results[key]["rmse_log_mean"], **timing[key]}  # Метрики сохранённых объектов.
metric_table = pd.DataFrame(metrics).T  # Итоговое сравнение.
display(metric_table[["rmse_log", "mae_log", "r2_log", "rmse_real", "mae_real", "r2_real", "fit_time_sec"]].round(4))  # Обе шкалы подписаны.
baseline_log = np.full(len(y_valid_log), y_train_log.mean())  # Наивный прогноз среднего по train.
print("Baseline RMSE(log):", np.sqrt(mean_squared_error(y_valid_log, baseline_log)), "R²(log):", r2_score(y_valid_log, baseline_log))  # Проверяем полезность моделей.
best_log = valid_predictions[best_key]["log"]  # Прогноз модели, выбранной заранее по CV.
best_real = valid_predictions[best_key]["real"]  # Та же модель в долларах.
residual_real = y_valid_real.to_numpy() - best_real  # Положительный остаток означает недооценку.
residual_log = y_valid_log.to_numpy() - best_log  # Остатки в шкале обучения.
residual_summary = {"mean_usd": float(residual_real.mean()), "std_usd": float(residual_real.std(ddof=1)),
                    "mean_log": float(residual_log.mean()), "max_abs_usd": float(np.abs(residual_real).max())}  # Фактическая диагностика.
print("Остатки выбранной модели:", residual_summary)  # Не утверждаем заранее нулевое смещение.
pd.DataFrame({"Id": valid_frame["Id"].to_numpy(), "actual_usd": y_valid_real.to_numpy(),
              **{key + "_log": p["log"] for key, p in valid_predictions.items()},
              **{key + "_usd": p["real"] for key, p in valid_predictions.items()}}).to_csv("examples/validation_predictions.csv", index=False)  # Сохраняем все holdout-прогнозы.

# %% Кодовая ячейка 17
fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))  # Проверяем предсказания в двух шкалах.
for ax, actual, predicted, label in [(axes[0], y_valid_real, best_real, "USD"), (axes[1], y_valid_log, best_log, "log(1 + USD)")]:
    ax.scatter(actual, predicted, alpha=.55, s=20, color="#4b83b4")  # Все отложенные дома.
    low, high = min(np.min(actual), np.min(predicted)), max(np.max(actual), np.max(predicted))  # Общие пределы сравнения.
    ax.plot([low, high], [low, high], "--", color="#b75c59")  # Диагональ идеального прогноза.
    ax.set(xlabel=f"Фактическая цена, {label}", ylabel=f"Прогноз, {label}", title=f"{model_names[best_key]}: holdout")  # Выбранная модель явно названа.
fig.tight_layout()  # Сохраняем подписи обеих панелей.
fig.savefig("examples/predictions_scatter.png", bbox_inches="tight")  # Обязательная диаграмма предсказаний.
plt.show()  # Включаем сравнение в ноутбук.
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))  # Остатки в долларах, логарифмах и их распределение.
axes[0].scatter(best_real, residual_real, s=20, alpha=.5)  # Проверяем дорогие объекты.
axes[0].axhline(0, color="#b75c59", linestyle="--")  # Нулевая ошибка.
axes[0].set(title="Остатки в USD", xlabel="Предсказанная цена, USD", ylabel="Факт − прогноз, USD")  # Смысл знака указан.
axes[1].scatter(best_log, residual_log, s=20, alpha=.5)  # Диагностика шкалы обучения.
axes[1].axhline(0, color="#b75c59", linestyle="--")  # Опорная линия.
axes[1].set(title="Остатки в log-шкале", xlabel="Прогноз log(1 + USD)", ylabel="Остаток log")  # Не смешиваем денежные и log-ошибки.
sns.histplot(residual_log, bins=25, kde=True, ax=axes[2], color="#4d9a83")  # Форма ошибок, не самой цены.
axes[2].set(title="Распределение log-остатков", xlabel="Остаток log", ylabel="Число домов")  # Читаемые оси.
fig.tight_layout()  # Не допускаем наложения подписей.
fig.savefig("examples/residuals.png", bbox_inches="tight")  # Обязательный анализ остатков.
plt.show()  # Сохраняем выполненные графики.
fig, ax = plt.subplots(figsize=(14, 3.1))  # Отдельное изображение метрик для LMS.
ax.axis("off")  # Таблица не нуждается в координатных осях.
display_rows = [[model_names[k], f"{m['rmse_log']:.4f}", f"{m['mae_log']:.4f}", f"{m['r2_log']:.4f}", f"{m['rmse_real']:,.0f}", f"{m['mae_real']:,.0f}", f"{m['r2_real']:.4f}"] for k, m in metrics.items()]  # Одинаковые цифры с JSON.
table = ax.table(cellText=display_rows, colLabels=["Модель", "RMSE log", "MAE log", "R² log", "RMSE USD", "MAE USD", "R² USD"], colWidths=[.23, .12, .12, .12, .14, .14, .12], cellLoc="center", loc="center")  # Все обязательные метрики.
table.auto_set_font_size(False)  # Управляем читаемым шрифтом.
table.set_fontsize(9)  # Семь столбцов помещаются без обрезки.
table.scale(1, 1.8)  # Увеличиваем межстрочные интервалы.
ax.set_title("House Prices: 292 отложенных дома, random_state=42", pad=12)  # Указываем условия проверки.
fig.tight_layout()  # Экспорт без обрезанных заголовков.
fig.savefig("examples/metrics_table.png", bbox_inches="tight")  # Изображение для самостоятельной сдачи.
plt.show()  # Та же таблица видна в отчёте.

# %% Кодовая ячейка 20
ridge_model = fitted_pipelines["ridge"].named_steps["regressor"]  # Регуляризованная модель для интерпретации.
lasso_model = fitted_pipelines["lasso"].named_steps["regressor"]  # Проверяем разреженность коэффициентов.
coefficient_frame = pd.DataFrame({"Feature": feature_cols,
    **{k: fitted_pipelines[k].named_steps["regressor"].coef_ for k in ["linear_regression", "ridge", "lasso", "elasticnet"]}})  # Сравнимые входные схемы.
ridge_top = coefficient_frame.assign(abs_coef=np.abs(coefficient_frame["ridge"])).nlargest(20, "abs_coef").sort_values("ridge")  # Выбираем по модулю, а не только по положительному знаку.
fig, ax = plt.subplots(figsize=(11, 7.5))  # Двадцать полных имён признаков.
ax.barh(ridge_top["Feature"], ridge_top["ridge"], color=["#4d9a83" if v > 0 else "#b75c59" for v in ridge_top["ridge"]])  # Знак отражён цветом.
ax.axvline(0, color="black", linewidth=.8)  # Нулевая условная связь.
ax.set(title="Ridge: 20 крупнейших коэффициентов по модулю", xlabel="Коэффициент стандартизованного признака для log-цели", ylabel="Признак")  # Указываем масштаб.
fig.tight_layout()  # Имена One-Hot столбцов не обрезаются.
fig.savefig("examples/coefficients.png", bbox_inches="tight")  # Сохраняем интерпретацию.
plt.show()  # Включаем рисунок в отчёт.
display(ridge_top[["Feature", "ridge"]])  # Даём численные значения.
n_zero = int((lasso_model.coef_ == 0).sum())  # Считаем точные нули L1-регуляризации.
print("Lasso занулил", n_zero, "из", len(feature_cols), "коэффициентов")  # Нули не равны доказанной незначимости признаков.
display(coefficient_frame.loc[coefficient_frame["lasso"] != 0].sort_values("lasso", ascending=False).head(15))  # Положительные отобранные веса.
coefficient_frame.to_csv("examples/coefficients.csv", index=False)  # Полные коэффициенты четырёх моделей.
corr = train_df[["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "SalePrice"]].corr()  # Описательные связи исходных данных.
quality_summary = train_df.groupby("OverallQual")["SalePrice"].agg(["count", "median", "mean"])  # Состав групп и типичные цены.
pool_summary = train_df.assign(HasPool=train_df["PoolArea"] > 0).groupby("HasPool")["SalePrice"].agg(["count", "median", "mean"])  # Проверяем редкость бассейнов.
display(quality_summary)  # Величина связи не подменяется произвольной надбавкой в процентах.
display(pool_summary)  # Маленькая группа требует осторожности в обобщении.
interpretation = {"n_features": len(feature_cols), "lasso_zero": n_zero,
    "ridge_positive": coefficient_frame.nlargest(5, "ridge")[["Feature", "ridge"]].to_dict("records"),
    "ridge_negative": coefficient_frame.nsmallest(5, "ridge")[["Feature", "ridge"]].to_dict("records"),
    "quality_price_corr": float(corr.loc["OverallQual", "SalePrice"]),
    "area_price_corr": float(corr.loc["GrLivArea", "SalePrice"]),
    "garage_corr": float(corr.loc["GarageCars", "GarageArea"]),
    "pool": {str(k): {"n": int(r["count"]), "median": float(r["median"]), "mean": float(r["mean"])} for k, r in pool_summary.iterrows()}}  # Числа для текстовых выводов.
Path("models/interpretation.json").write_text(json.dumps(interpretation, ensure_ascii=False, indent=2), encoding="utf-8")  # Сохраняем основания выводов.

# %% Кодовая ячейка 23
template_index = (np.log1p(development["SalePrice"]) - y_train_log.median()).abs().idxmin()  # Выбираем основу только из development.
template_values = development.loc[template_index].drop(labels=["Id", "SalePrice"]).to_dict()  # Истинная цена не попадает в профиль.
template_house = {k: None if pd.isna(v) else v for k, v in template_values.items()}  # JSON содержит null вместо нечислового NaN.
specifications = [(9, 2800, 4600, 2015, 3, 3.5), (6, 1500, 2500, 2000, 2, 2.0),
                  (4, 900, 1500, 1950, 1, 1.0), (10, 4000, 6000, 2018, 4, 4.0),
                  (5, 1200, 1800, 1985, 1, 1.5)]  # Пять обязательных сочетаний параметров.
demo_houses, demo_rows = [], []  # Сохраняем полные профили для повторной загрузки моделей.
for number, (quality, area, total, built, garage, bath) in enumerate(specifications, 1):
    house = template_house.copy()  # У всех сценариев одна общая контролируемая основа.
    second = int(area * .35) if area > 2000 else 0  # Задаём правдоподобное распределение этажей.
    full_bath = int(bath) - (1 if bath >= 3 else 0)  # Часть ванных располагаем в подвале.
    house.update(OverallQual=quality, GrLivArea=area, YearBuilt=built, YearRemodAdd=built,
        YrSold=max(2010, built + 1), GarageCars=garage, GarageArea=garage * 240, GarageYrBlt=built,
        TotalBsmtSF=total-area, BsmtFinSF1=0, BsmtFinSF2=0, BsmtUnfSF=total-area,
        FullBath=full_bath, HalfBath=int(bath % 1 != 0), BsmtFullBath=1 if bath >= 3 else 0, BsmtHalfBath=0,
        **{"1stFlrSF": area-second, "2ndFlrSF": second, "LowQualFinSF": 0})  # Все агрегаты считаются из согласованных сырых полей.
    demo_houses.append(house)  # Эти же словари используются при проверке восстановления.
    row = {"Пример": number, "OverallQual": quality, "GrLivArea": area, "TotalSF": total,
           "YearBuilt": built, "GarageCars": garage, "TotalBath": bath}  # Показываем ключевые входы задания.
    for key, pipeline in fitted_pipelines.items():
        row[model_names[key]] = predict_price(house, pipeline.named_steps["regressor"], pipeline.named_steps["scaler"],
            feature_cols, pipeline.named_steps["preprocessor"])  # Каждая модель получает собственные сохранённые преобразования.
    demo_rows.append(row)  # Цены остаются числовыми до форматирования вывода.
demo_results = pd.DataFrame(demo_rows)  # Пять строк и прогнозы всех пяти моделей.
display(demo_results.round(0))  # Все цены выражены в долларах.
demo_results.to_csv("examples/prediction_examples.csv", index=False)  # Отдельная таблица сценариев.
Path("examples/demo_houses.json").write_text(json.dumps(demo_houses, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # Сохраняем полные входы.
RUN_INTERACTIVE = False  # Автоматическое выполнение не должно ожидать клавиатуру.
if RUN_INTERACTIVE:
    interactive_predict({"best_model": best_model, "scaler": scaler, "feature_cols": feature_cols, "preprocessor": preprocessor})  # Тот же инференс.

# %% Кодовая ячейка 26
artifact_names = {"linear_regression": "lr_model", "ridge": "ridge_model", "lasso": "lasso_model",
                  "elasticnet": "enet_model", "gradient_boosting": "gbr_model"}  # Имена из структуры задания.
for key, name in artifact_names.items():
    joblib.dump(fitted_pipelines[key].named_steps["regressor"], f"models/{name}.pkl", compress=3)  # Все пять моделей отдельно.
    other_prep = fitted_pipelines[key].named_steps["preprocessor"]  # Проверяем согласованность входов разных estimator.
    assert list(other_prep.get_feature_names_out()) == feature_cols  # Одна схема после fit на одинаковом development.
    np.testing.assert_allclose(fitted_pipelines[key].named_steps["scaler"].mean_, scaler.mean_)  # Скейлеры совпадают.
for name, obj in {"best_model": best_model, "best_pipeline": best_pipeline, "scaler": scaler, "preprocessor": preprocessor}.items():
    joblib.dump(obj, f"models/{name}.pkl", compress=3)  # Сохраняем единую логику сырых входов.
Path("models/feature_cols.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")  # Порядок для инференса.
Path("models/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")  # Фактические оценки всех моделей.
Path("models/cv_results.json").write_text(json.dumps(cv_results, indent=2), encoding="utf-8")  # Основание выбора без holdout.
metadata = {"training_date": config["date"], "student": config["student"], "group": config["group"], "teacher": config["teacher"],
    "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "pandas_version": pd.__version__,
    "random_state": 42, "cv_folds": 5, "selection_metric": "train_CV_RMSE_log", "selected_model": best_key,
    "outer_train_ids": [int(i) for i in outer_train_ids], "training_ids": development["Id"].astype(int).tolist(),
    "validation_ids": [int(i) for i in valid_ids], "removed_ids": removed_ids, "iqr_flags": int(iqr_flag.sum()), "iqr_bounds": [float(lower), float(upper)],
    "training_rows": len(X_train), "validation_rows": len(X_valid), "feature_cols": feature_cols,
    "log_predictors": preprocessor.log_cols_, "price_stats": price_stats, "residuals": residual_summary,
    "warnings": training_warnings, "data_sources": json.loads(Path("data/sources.json").read_text()),
    "repository_url": config["repository_url"], "github_raw_verified": False,
    "ai_assistance": "OpenAI Codex: код, пояснения, проверки"}  # Полная воспроизводимость.
Path("models/metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")  # Метаданные без фиктивного статуса публикации.
test_predictions = np.expm1(best_pipeline.predict(test_df.drop(columns="Id")))  # Test не фильтруется и не участвует в fit.
assert len(test_predictions) == 1459 and np.isfinite(test_predictions).all() and (test_predictions > 0).all()  # Формат Kaggle.
submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": test_predictions})  # Столбцы в нужном порядке.
submission.to_csv("submission.csv", index=False)  # Готовый файл прогнозов.
display(submission.head())  # Проверяем единицы и ID.
print("Сохранено прогнозов:", len(submission), "Выбрана модель:", model_names[best_key])  # Итог пакетного инференса.

# %% Кодовая ячейка 29
loaded = load_local_artifacts("models")  # Восстанавливаем объекты без обучения.
reload_results = []  # Три профиля для каждой из пяти моделей.
for number, house in enumerate(demo_houses[:3], 1):
    for key, name in artifact_names.items():
        expected = float(demo_results.iloc[number - 1][model_names[key]])  # Прогноз раздела 9.
        actual = predict_price(house, loaded[name], loaded["scaler"], loaded["feature_cols"], loaded["preprocessor"])  # Сохранённые объекты.
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-7)  # Проверяем численное совпадение.
        reload_results.append({"Пример": number, "Модель": model_names[key], "Цена USD": actual, "Совпадает": True})  # Протокол 15 сравнений.
display(pd.DataFrame(reload_results).round(2))  # Результаты проверки восстановления.
config = json.loads(Path("student.json").read_text(encoding="utf-8"))  # Подхватываем адрес после самостоятельной публикации.
github_verified = False  # Не объявляем невыполненную сетевую проверку успешной.
if config.get("repository_url") and config.get("run_github_check", False):
    remote = load_artifacts_from_github(config["repository_url"], config.get("branch", "main"))  # Настоящие raw HTTP-запросы.
    for house in demo_houses[:3]:
        expected = predict_price(house, loaded["best_model"], loaded["scaler"], loaded["feature_cols"], loaded["preprocessor"])
        actual = predict_price(house, remote["best_model"], remote["scaler"], remote["feature_cols"], remote["preprocessor"])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-7)  # Проверяем загруженный комплект.
    github_verified = True  # Успех только после реальной загрузки и сравнения.
    print("GitHub raw проверен:", remote["base_url"])  # Показываем фактический источник.
else:
    print("Локальная загрузка проверена. GitHub raw не проверен: автор публикует проект самостоятельно.")  # Честный статус.
metadata.update(repository_url=config.get("repository_url"), github_raw_verified=github_verified)  # Сохраняем настоящий статус.
Path("models/metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")  # Обновляем метаданные.
