# Importação de bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
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

# Definir coluna label (Será usada para testar os resultados gerado pelo Modelo)
target_col = 'is_fraud'

# Remover colunas irrelevantes e label dos dados
columns_to_drop_final = ['Unnamed: 0', 'trans_num', 'first', 'last', 'street', 'city', 'zip', 'dob', 'ssn']
df_processed = dataframe.drop(columns=columns_to_drop_final, errors='ignore').copy()

# Separar X e y
X = df_processed.drop(columns=[target_col], errors='ignore')
y = df_processed[target_col]

# Definir colunas numéricas e categóricas
numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time',
                  'time_since_last_transaction', 'merch_lat', 'merch_long']
categorical_cols = ['category', 'gender', 'state', 'job', 'merchant']

# Dividir conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Calculo de contaminação
fraud_ratio = y_train.sum() / len(y_train)
contamination_rate = max(fraud_ratio, 0.01)

# Preprocessamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='drop'
)

# Treinamento do Modelo
iso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('anomaly_detector', IsolationForest(
        n_estimators=100,
        contamination=contamination_rate,
        random_state=42,
        n_jobs=-1
    ))
])

iso_pipeline.fit(X_train)

# Classificando de acordo com o modelo
X_test_preprocessed = iso_pipeline.named_steps['preprocessor'].transform(X_test)
anomaly_scores = -iso_pipeline.named_steps['anomaly_detector'].decision_function(X_test_preprocessed)

# Ajuste de limite (threshold)
threshold = np.percentile(anomaly_scores, 99.5)
iso_predictions = (anomaly_scores >= threshold).astype(int)
iso_predictions = 1 - iso_predictions


# Avaliação do Modelo
print("\nRelatório de Classificação (Isolation Forest):\n",
      classification_report(y_test, iso_predictions, target_names=['Não Fraude', 'Fraude']))

# Métricas
accuracy_iso = accuracy_score(y_test, iso_predictions)
precision_iso = precision_score(y_test, iso_predictions, pos_label=1)
recall_iso = recall_score(y_test, iso_predictions, pos_label=1)
f1_iso = f1_score(y_test, iso_predictions, pos_label=1)

iso_anomaly_scores = -iso_pipeline.named_steps['anomaly_detector'].decision_function(
    iso_pipeline.named_steps['preprocessor'].transform(X_test)
)

scaled_scores = MinMaxScaler().fit_transform(iso_anomaly_scores.reshape(-1, 1)).flatten()

roc_auc_iso = roc_auc_score(y_test, scaled_scores)
pr_auc_iso = average_precision_score(y_test, scaled_scores)

# Impressão dos resultados
print(f"Acurácia: {accuracy_iso:.4f}")
print(f"Precisão (Fraude): {precision_iso:.4f}")
print(f"Recall (Fraude): {recall_iso:.4f}")
print(f"F1-Score (Fraude): {f1_iso:.4f}")
print(f"AUC-ROC: {roc_auc_iso:.4f}")
print(f"AUC-PR: {pr_auc_iso:.4f}")