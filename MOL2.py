import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# ======================================================
# Этап 1. Подготовка данных (как в вашем первоначальном коде)
# ======================================================

# Чтение датасета
df = pd.read_csv('D:/для работ/MMOlab/titanic_data.csv')

# Отображение всех столбцов при выводе
pd.options.display.max_columns = None

print("Первые 10 строк датасета:")
print(df.head(10))

# Подсчёт пропущенных значений по столбцам
print("\nКоличество пропущенных значений по столбцам:")
print(df.isnull().sum())

# Заполнение пропущенных значений:
# Для числовых столбцов — медианой
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Для категориальных столбцов (тип object или category) — модой
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nПосле заполнения пропусков:")
print(df.isnull().sum())

# Нормализация числовых данных с помощью MinMaxScaler
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Преобразование категориальных данных в dummy-переменные с параметром drop_first=True
df = pd.get_dummies(df, drop_first=True)

print("\nДатасет после нормализации и преобразования категориальных данных:")
print(df.head())

# ======================================================
# Этап 2. Задача регрессии (прогноз одного из непрерывных признаков)
# ======================================================

# В данном примере в качестве целевого признака для регрессии выбран столбец 'Fare'
if 'Fare' not in df.columns:
    raise ValueError("Столбец 'Fare' не найден в датасете! Выберите другой непрерывный признак для регрессии.")

# Отделяем признаки от целевой переменной
X_reg = df.drop('Fare', axis=1)
y_reg = df['Fare']

# Разделяем данные на обучающую (80%) и тестовую (20%) выборки
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Создаем и обучаем модель линейной регрессии
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# Предсказываем значения на тестовой выборке
y_pred_reg = reg_model.predict(X_test_reg)

# Оценка модели: среднеквадратичная ошибка (MSE) и коэффициент детерминации (R²)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("\n=== Задача регрессии (прогноз 'Fare') ===")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Если результаты модели неудовлетворительны, можно попробовать:
# • Использовать нелинейные модели (например, RandomForestRegressor или GradientBoostingRegressor);
# • Добавить полиномиальные признаки или выполнить отбор признаков;
# • Провести поиск по гиперпараметрам (GridSearchCV или RandomizedSearchCV).

# ======================================================
# Этап 3. Задача классификации (например, прогноз выживания)
# ======================================================

# Для классификации в качестве целевого признака используется столбец 'Survived'
if 'Survived' not in df.columns:
    raise ValueError("Столбец 'Survived' не найден в датасете! Проверьте название целевого признака для классификации.")

# Отделяем признаки от целевой переменной
X_clf = df.drop('Survived', axis=1)
y_clf = df['Survived']

# Разбиваем данные на обучающую и тестовую выборки (80%/20%)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Создаем и обучаем модель логистической регрессии (для бинарной классификации)
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_clf, y_train_clf)

# Получаем предсказания на тестовой выборке
y_pred_clf = clf_model.predict(X_test_clf)

# Оцениваем модель: точность (accuracy) и подробный отчёт классификации
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print("\n=== Задача классификации (прогноз 'Survived') ===")
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test_clf, y_pred_clf))

