import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prepare_data():
    # Incarca datele
    df = pd.read_csv('./data/processed/unprepared_data.csv')
    
    # Trateaza valorile lipsa
    missing_vals = ['hours_studied_weekly', 'attendance_rate', 'previous_mark']
    
    # Completeaza valorile lipsa cu mediana
    num_imputer = SimpleImputer(strategy='median')
    df[missing_vals] = num_imputer.fit_transform(df[missing_vals])
    
    # Imparte datele
    train_df, test_df = train_test_split(
        df, test_size=0.285, random_state=42)
    
    # 4. Salveaza
    train_df.to_csv('./data/processed/train.csv', index=False)
    test_df.to_csv('./data/processed/test.csv', index=False)

if __name__ == "__main__":
    prepare_data()