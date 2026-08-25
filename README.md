# ML Playground: House Pricing & Credit Scoring

Учебный репозиторий с двумя end-to-end ML-проектами: регрессией цены недвижимости и бинарной классификацией кредитного риска. Оба ноутбука построены по единому пайплайну: EDA → отбор признаков → обучение нескольких моделей → сравнение по метрикам.

## Содержание

| Ноутбук | Задача | Тип ML-задачи | Целевая переменная |
|---|---|---|---|
| `kc_house.ipynb` | Прогноз стоимости дома | Регрессия | `price` |
| `credit.ipynb` | Прогноз кредитного дефолта | Бинарная классификация | `default_risk` |

---

## 1. `kc_house.ipynb` — предсказание цены недвижимости (King County)

**Датасет:** `kc_house_data.csv` — характеристики домов (площадь, этажность, состояние, координаты, год постройки и т.д.)

**Пайплайн:**
1. **EDA** — `df.info()`, `df.describe()`, проверка пропусков, распределение `price` (в т.ч. логарифмическое `log1p`), матрица корреляций (`sns.heatmap`), boxplot для выбросов.
2. **Предобработка** — извлечение года из `date`, удаление `id`, заполнение пропусков медианой (`SimpleImputer`).
3. **Отбор признаков** — `LassoCV` (L1-регуляризация) для отбора ненулевых коэффициентов.
4. **Разбиение выборки** — `train_test_split` (70/30).
5. **Обучение моделей:**
   - `LinearRegression`
   - `RandomForestRegressor` (подбор гиперпараметров через `GridSearchCV`: `n_estimators`, `max_depth`, `min_samples_split`)
   - `CatBoostRegressor` (500 итераций, `learning_rate=0.1`)
6. **Валидация** — `cross_val_score` (5-fold) на RMSE.
7. **Метрики:** MAE, MSE, RMSE, R².
8. **Диагностика** — графики "предсказание vs истина" и анализ остатков для каждой модели.

**Итог:** лучшая модель — **Random Forest** (R² = 0.79, MAE ≈ 92.9K), CatBoost — близко (R² = 0.77), линейная регрессия заметно слабее (R² = 0.55).

---

## 2. `credit.ipynb` — кредитный скоринг (predictive risk of default)

**Датасет:** `credit_data.csv` — признаки заёмщика, целевая переменная `default_risk` (бинарная: просрочка / нет просрочки).

**Пайплайн:**
1. **EDA** — `df.info()`, `df.describe()`, корреляция признаков с таргетом.
2. **Разбиение выборки** — `train_test_split` со `stratify=y` (сохранение баланса классов), 80/20.
3. **Обучение моделей** через единую функцию `train_and_evaluate()`:
   - `LogisticRegression` (на стандартизированных признаках, `StandardScaler`)
   - `DecisionTreeClassifier` (`max_depth=6`)
   - `RandomForestClassifier` (`n_estimators=200`, `max_depth=8`)
4. **Метрики:** Accuracy, Precision, Recall, F1-score, ROC-AUC.
5. **Визуализация:** Confusion Matrix (по каждой модели) и общий график ROC-кривых для сравнения.
6. **Сводная таблица** — метрики всех моделей в едином `DataFrame`.

**Итог:** лучшая модель — **Logistic Regression**.

---

## Технологии и библиотеки

- **Данные / визуализация:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`
- **ML:** `scikit-learn` (`LinearRegression`, `LogisticRegression`, `LassoCV`, `DecisionTreeClassifier`, `RandomForestRegressor`, `RandomForestClassifier`, `GridSearchCV`, `cross_val_score`, `StandardScaler`, `SimpleImputer`), `catboost`
- **Метрики:** MAE, MSE, RMSE, R² (регрессия); Accuracy, Precision, Recall, F1, ROC-AUC (классификация)
