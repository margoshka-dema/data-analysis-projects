# ML Playground: House Pricing & Credit Scoring

Учебный репозиторий с двумя end-to-end ML-проектами: регрессией цены недвижимости и бинарной классификацией кредитного риска. Оба ноутбука построены по единому пайплайну: EDA → отбор признаков → обучение нескольких моделей → сравнение по метрикам.

## Содержание

| Ноутбук | Задача | Тип ML-задачи | Целевая переменная |
|---|---|---|---|
| `credit.ipynb` | Прогноз кредитного дефолта | Бинарная классификация | `default_risk` |

---

## `credit.ipynb` — кредитный скоринг (predictive risk of default)

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
- **ML:** `scikit-learn` (`LinearRegression`, `LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `StandardScaler`, `SimpleImputer`), `catboost`
- **Метрики:** Accuracy, Precision, Recall, F1, ROC-AUC (классификация)
