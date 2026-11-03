# Online Shoppers Purchasing Intention

Sakar & Kastro (2018), UCI, DOI 10.24432/C5F88Q, CC BY 4.0.

Исходно 12330 строк, 17 входных полей и Revenue. 1908 покупок, 10422 без покупки. 125 точных повторов исключены перед разбиением.

|                         | dtype   |   missing |
|:------------------------|:--------|----------:|
| Administrative          | int64   |         0 |
| Administrative_Duration | float64 |         0 |
| Informational           | int64   |         0 |
| Informational_Duration  | float64 |         0 |
| ProductRelated          | int64   |         0 |
| ProductRelated_Duration | float64 |         0 |
| BounceRates             | float64 |         0 |
| ExitRates               | float64 |         0 |
| PageValues              | float64 |         0 |
| SpecialDay              | float64 |         0 |
| Month                   | object  |         0 |
| OperatingSystems        | int64   |         0 |
| Browser                 | int64   |         0 |
| Region                  | int64   |         0 |
| TrafficType             | int64   |         0 |
| VisitorType             | object  |         0 |
| Weekend                 | bool    |         0 |
| Revenue                 | bool    |         0 |


## Поля исходного набора

| Поле | Тип CSV | Описание | Пропуски |
|---|---|---|---:|
| Administrative | int64 | Число административных страниц | 0 |
| Administrative_Duration | float64 | Секунды на административных страницах | 0 |
| Informational | int64 | Число информационных страниц | 0 |
| Informational_Duration | float64 | Секунды на информационных страницах | 0 |
| ProductRelated | int64 | Число товарных страниц | 0 |
| ProductRelated_Duration | float64 | Секунды на товарных страницах | 0 |
| BounceRates | float64 | Показатель отказов | 0 |
| ExitRates | float64 | Показатель выходов | 0 |
| PageValues | float64 | Средняя ценность посещённых страниц по данным набора | 0 |
| SpecialDay | float64 | Близость особого дня, 0–1 | 0 |
| Month | object | Месяц; в исходном CSV June | 0 |
| OperatingSystems | int64 | Код ОС | 0 |
| Browser | int64 | Код браузера | 0 |
| Region | int64 | Код региона | 0 |
| TrafficType | int64 | Код источника трафика | 0 |
| VisitorType | object | New_Visitor, Returning_Visitor или Other | 0 |
| Weekend | bool | Сессия в выходной день | 0 |
| Revenue | bool | Целевая: True означает покупку | 0 |
