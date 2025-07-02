# Importação de bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# Importação e visualização das colunas
dataframe = pd.read_csv('/kaggle/input/fraud-detection/fraudTrain.csv')
dataframe.head()

# Datas para Objetos Datetimes
dataframe['trans_date_trans_time'] = pd.to_datetime(dataframe['trans_date_trans_time'])
dataframe['hour_of_day'] = dataframe['trans_date_trans_time'].dt.hour
dataframe['day_of_week'] = dataframe['trans_date_trans_time'].dt.dayofweek
dataframe['month'] = dataframe['trans_date_trans_time'].dt.month

# Diferença de tempo entre transações.
dataframe = dataframe.sort_values(by=['cc_num', 'trans_date_trans_time'])
dataframe['time_since_last_transaction'] = dataframe.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds().fillna(0)

# Definir coluna label
target_col = 'is_fraud'
# Removendo colunas irrelevantes e coluna label dos dados de treinamento
columns_to_drop_final = ['Unnamed: 0', 'trans_num', 'first', 'last', 'street', 'city', 'zip', 'dob', 'ssn']
df_processed = dataframe.drop(columns=columns_to_drop_final, errors='ignore').copy()
X = df_processed.drop(columns=[target_col], errors='ignore')
y = df_processed[target_col]
# Categorização de colunas
numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'time_since_last_transaction', 'merch_lat', 'merch_long']
categorical_cols = ['category', 'gender', 'state', 'job', 'merchant']

# Pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='drop'
)

# Divisão de dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Treinamento
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))])

rf_pipeline.fit(X_train, y_train)
rf_predictions_cpu = rf_pipeline.predict(X_test)
rf_probabilities_cpu = rf_pipeline.predict_proba(X_test)[:, 1]
rf_predictions = (rf_probabilities_cpu >= 0.3).astype(int)

print("\nRelatório de Classificação:\n", classification_report(y_test, rf_predictions_cpu, target_names=['Não Fraude', 'Fraude']))

accuracy_rf = accuracy_score(y_test, rf_predictions_cpu)
precision_rf = precision_score(y_test, rf_predictions_cpu, pos_label=1)
recall_rf = recall_score(y_test, rf_predictions_cpu, pos_label=1)
f1_rf = f1_score(y_test, rf_predictions_cpu, pos_label=1)
roc_auc_rf = roc_auc_score(y_test, rf_probabilities_cpu)
pr_auc_rf = average_precision_score(y_test, rf_probabilities_cpu)

print(f"Acurácia: {accuracy_rf:.4f}")
print(f"Precisão (Fraude): {precision_rf:.4f}")
print(f"Recall (Fraude): {recall_rf:.4f}")
print(f"F1-Score (Fraude): {f1_rf:.4f}")
print(f"AUC-ROC: {roc_auc_rf:.4f}")
print(f"AUC-PR: {pr_auc_rf:.4f}")