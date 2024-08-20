import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_paths):
    data = {}
    for key, path in file_paths.items():
        data[key] = pd.read_csv(path)
    return data

def preprocess_data(data):
    processed_data = {}
    for key, df in data.items():
        features = df.drop(columns='target')
        target = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        processed_data[key] = (X_train, X_test, y_train, y_test)
    return processed_data
