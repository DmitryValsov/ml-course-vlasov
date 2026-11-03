# Telco Customer Churn

7043 клиента, 19 входных признаков, customerID и Churn.

No: 5174 (73.46%); Yes: 1869 (26.54%). tenure — месяцы; charges — денежные суммы по данным IBM.

|                  | dtype   |   missing |
|:-----------------|:--------|----------:|
| customerID       | object  |         0 |
| gender           | object  |         0 |
| SeniorCitizen    | int64   |         0 |
| Partner          | object  |         0 |
| Dependents       | object  |         0 |
| tenure           | int64   |         0 |
| PhoneService     | object  |         0 |
| MultipleLines    | object  |         0 |
| InternetService  | object  |         0 |
| OnlineSecurity   | object  |         0 |
| OnlineBackup     | object  |         0 |
| DeviceProtection | object  |         0 |
| TechSupport      | object  |         0 |
| StreamingTV      | object  |         0 |
| StreamingMovies  | object  |         0 |
| Contract         | object  |         0 |
| PaperlessBilling | object  |         0 |
| PaymentMethod    | object  |         0 |
| MonthlyCharges   | float64 |         0 |
| TotalCharges     | object  |        11 |
| Churn            | object  |         0 |

TotalCharges: 11 пробелов преобразуются в NaN и заполняются медианой train.



## Описание полей

| Поле | Тип CSV | Описание |
|---|---|---|
| customerID | object | Идентификатор клиента, исключён из X |
| gender | object | Пол |
| SeniorCitizen | int64 | Бинарный признак пожилого клиента |
| Partner | object | Наличие партнёра |
| Dependents | object | Наличие иждивенцев |
| tenure | int64 | Месяцы обслуживания |
| PhoneService | object | Телефонная услуга |
| MultipleLines | object | Несколько телефонных линий |
| InternetService | object | Тип интернета |
| OnlineSecurity | object | Онлайн-безопасность |
| OnlineBackup | object | Резервное копирование |
| DeviceProtection | object | Защита устройства |
| TechSupport | object | Техподдержка |
| StreamingTV | object | Стриминг ТВ |
| StreamingMovies | object | Стриминг фильмов |
| Contract | object | Тип договора |
| PaperlessBilling | object | Безбумажные счета |
| PaymentMethod | object | Способ оплаты |
| MonthlyCharges | float64 | Ежемесячный платёж |
| TotalCharges | object | Накопленные платежи; 11 пробельных значений |
| Churn | object | Ушёл Yes=1, остался No=0 |
