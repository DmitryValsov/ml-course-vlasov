# House Prices: описание набора

Источник: Kaggle; учебная копия D2L, URLs и SHA-256 в sources.json.

Train: 1460×81, test: 1459×80. 79 входных признаков, Id и SalePrice. Площади в sq ft, цены в USD.

|               | dtype   |   missing_count |   missing_pct |
|:--------------|:--------|----------------:|--------------:|
| Id            | int64   |               0 |         0     |
| MSSubClass    | int64   |               0 |         0     |
| MSZoning      | object  |               0 |         0     |
| LotFrontage   | float64 |             259 |        17.74  |
| LotArea       | int64   |               0 |         0     |
| Street        | object  |               0 |         0     |
| Alley         | object  |            1369 |        93.767 |
| LotShape      | object  |               0 |         0     |
| LandContour   | object  |               0 |         0     |
| Utilities     | object  |               0 |         0     |
| LotConfig     | object  |               0 |         0     |
| LandSlope     | object  |               0 |         0     |
| Neighborhood  | object  |               0 |         0     |
| Condition1    | object  |               0 |         0     |
| Condition2    | object  |               0 |         0     |
| BldgType      | object  |               0 |         0     |
| HouseStyle    | object  |               0 |         0     |
| OverallQual   | int64   |               0 |         0     |
| OverallCond   | int64   |               0 |         0     |
| YearBuilt     | int64   |               0 |         0     |
| YearRemodAdd  | int64   |               0 |         0     |
| RoofStyle     | object  |               0 |         0     |
| RoofMatl      | object  |               0 |         0     |
| Exterior1st   | object  |               0 |         0     |
| Exterior2nd   | object  |               0 |         0     |
| MasVnrType    | object  |             872 |        59.726 |
| MasVnrArea    | float64 |               8 |         0.548 |
| ExterQual     | object  |               0 |         0     |
| ExterCond     | object  |               0 |         0     |
| Foundation    | object  |               0 |         0     |
| BsmtQual      | object  |              37 |         2.534 |
| BsmtCond      | object  |              37 |         2.534 |
| BsmtExposure  | object  |              38 |         2.603 |
| BsmtFinType1  | object  |              37 |         2.534 |
| BsmtFinSF1    | int64   |               0 |         0     |
| BsmtFinType2  | object  |              38 |         2.603 |
| BsmtFinSF2    | int64   |               0 |         0     |
| BsmtUnfSF     | int64   |               0 |         0     |
| TotalBsmtSF   | int64   |               0 |         0     |
| Heating       | object  |               0 |         0     |
| HeatingQC     | object  |               0 |         0     |
| CentralAir    | object  |               0 |         0     |
| Electrical    | object  |               1 |         0.068 |
| 1stFlrSF      | int64   |               0 |         0     |
| 2ndFlrSF      | int64   |               0 |         0     |
| LowQualFinSF  | int64   |               0 |         0     |
| GrLivArea     | int64   |               0 |         0     |
| BsmtFullBath  | int64   |               0 |         0     |
| BsmtHalfBath  | int64   |               0 |         0     |
| FullBath      | int64   |               0 |         0     |
| HalfBath      | int64   |               0 |         0     |
| BedroomAbvGr  | int64   |               0 |         0     |
| KitchenAbvGr  | int64   |               0 |         0     |
| KitchenQual   | object  |               0 |         0     |
| TotRmsAbvGrd  | int64   |               0 |         0     |
| Functional    | object  |               0 |         0     |
| Fireplaces    | int64   |               0 |         0     |
| FireplaceQu   | object  |             690 |        47.26  |
| GarageType    | object  |              81 |         5.548 |
| GarageYrBlt   | float64 |              81 |         5.548 |
| GarageFinish  | object  |              81 |         5.548 |
| GarageCars    | int64   |               0 |         0     |
| GarageArea    | int64   |               0 |         0     |
| GarageQual    | object  |              81 |         5.548 |
| GarageCond    | object  |              81 |         5.548 |
| PavedDrive    | object  |               0 |         0     |
| WoodDeckSF    | int64   |               0 |         0     |
| OpenPorchSF   | int64   |               0 |         0     |
| EnclosedPorch | int64   |               0 |         0     |
| 3SsnPorch     | int64   |               0 |         0     |
| ScreenPorch   | int64   |               0 |         0     |
| PoolArea      | int64   |               0 |         0     |
| PoolQC        | object  |            1453 |        99.521 |
| Fence         | object  |            1179 |        80.753 |
| MiscFeature   | object  |            1406 |        96.301 |
| MiscVal       | int64   |               0 |         0     |
| MoSold        | int64   |               0 |         0     |
| YrSold        | int64   |               0 |         0     |
| SaleType      | object  |               0 |         0     |
| SaleCondition | object  |               0 |         0     |
| SalePrice     | int64   |               0 |         0     |

Статистики цели:
{
  "mean": 180921.19589041095,
  "median": 163000.0,
  "skew": 1.8828757597682129,
  "kurtosis": 6.536281860064529,
  "log_skew": 0.12134661989685329
}

## Основные поля

| Поле | Тип исходного train | Смысл | Пропуски train |
|---|---|---|---:|
| Id | int64 | Идентификатор дома; исключён из X | 0.00% |
| SalePrice | int64 | Цена продажи в исторических USD; целевая | 0.00% |
| OverallQual | int64 | Общее качество материалов и отделки, 1–10 | 0.00% |
| OverallCond | int64 | Общее состояние, 1–10 | 0.00% |
| GrLivArea | int64 | Жилая площадь над землёй, кв. футы | 0.00% |
| LotArea | int64 | Площадь участка, кв. футы | 0.00% |
| LotFrontage | float64 | Длина фасада участка, футы | 17.74% |
| Neighborhood | object | Район Ames | 0.00% |
| YearBuilt | int64 | Год постройки | 0.00% |
| YearRemodAdd | int64 | Год реконструкции | 0.00% |
| YrSold | int64 | Год продажи | 0.00% |
| TotalBsmtSF | int64 | Общая площадь подвала, кв. футы | 0.00% |
| 1stFlrSF | int64 | Площадь первого этажа, кв. футы | 0.00% |
| 2ndFlrSF | int64 | Площадь второго этажа, кв. футы | 0.00% |
| GarageCars | int64 | Вместимость гаража, автомобилей | 0.00% |
| GarageArea | int64 | Площадь гаража, кв. футы | 0.00% |
| ExterQual | object | Порядковая оценка внешнего качества | 0.00% |
| BsmtQual | object | Порядковая оценка качества/высоты подвала | 2.53% |
| PoolArea | int64 | Площадь бассейна, кв. футы | 0.00% |
