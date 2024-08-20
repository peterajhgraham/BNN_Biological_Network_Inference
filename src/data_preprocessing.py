import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_paths):
    data = {}
    for name, path in file_paths.items():
        data[name] = pd.read_csv(path)
    return data

def preprocess_data(data):
    processed_data = {}
    for key, df in data.items():
        # Example preprocessing steps
        if 'interaction_score' in df.columns:
            X = df.drop(columns=['interaction_score'])
            y = df['interaction_score']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            processed_data[key] = (X_train, X_test, y_train, y_test)
    return processed_data
