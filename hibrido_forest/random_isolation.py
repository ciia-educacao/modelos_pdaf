# Manipulação de dados
import numpy as np
import pandas as pd

# Pré-processamento e transformação
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
# Modelos
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline
# Divisão dos dados
from sklearn.model_selection import train_test_split
# Métricas de avaliação
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
# Visualização opcional (se quiser ver distribuição dos scores)
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
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='drop'
)

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pré-filtragem com Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
X_train_proc = preprocessor.fit_transform(X_train)
iso.fit(X_train_proc)

# Aplicar no conjunto de teste
X_test_proc = preprocessor.transform(X_test)
anomaly_scores = -iso.decision_function(X_test_proc)

# Selecionar apenas os mais anômalos (top 2%)
threshold = np.percentile(anomaly_scores, 95)
mask_suspects = anomaly_scores >= threshold

# Filtrar para Random Forest
X_test_suspects = X_test[mask_suspects]
y_test_suspects = y_test[mask_suspects]

# Treinar Random Forest no conjunto completo (poderia ser refinado também)
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
])

rf_pipeline.fit(X_train, y_train)

# Aplicar RF apenas nos casos suspeitos

rf_predictions = rf_pipeline.predict(X_test_suspects)
rf_probabilities = rf_pipeline.predict_proba(X_test_suspects)[:, 1]
rf_predictions = (rf_probabilities >= 0.3).astype(int)

# Avaliação
print("\nAvaliação apenas nos casos suspeitos (filtrados pelo Isolation Forest):")
print(classification_report(y_test_suspects, rf_predictions, target_names=['Não Fraude', 'Fraude']))

accuracy = accuracy_score(y_test_suspects, rf_predictions)
precision = precision_score(y_test_suspects, rf_predictions, pos_label=1)
recall = recall_score(y_test_suspects, rf_predictions, pos_label=1)
f1 = f1_score(y_test_suspects, rf_predictions, pos_label=1)
roc_auc = roc_auc_score(y_test_suspects, rf_probabilities)
pr_auc = average_precision_score(y_test_suspects, rf_probabilities)

print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão (Fraude): {precision:.4f}")
print(f"Recall (Fraude): {recall:.4f}")
print(f"F1-Score (Fraude): {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"AUC-PR: {pr_auc:.4f}")