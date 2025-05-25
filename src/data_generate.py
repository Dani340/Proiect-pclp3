# src/1_generate_data.py
import numpy as np
import pandas as pd
import random

def generate_student_data(n_samples=2000):
    np.random.seed(42)
    random.seed(42)
    
    data = {
        # 1. Gen (sir de caractere)
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.51, 0.49]),
        
        # 2. Facultate (val categoriala)
        'faculty': np.random.choice(
            ["Computer Science", "Mathematics", "Physics", "Biology", "Economics"], 
            n_samples,
            p=[0.3, 0.2, 0.2, 0.2, 0.1]
        ),
        
        # 3. Varsta (numar intreg)
        'age': np.random.randint(18, 25, n_samples),
        
        # 4. Bursa (val true/false)
        'has_scholarship': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        
        # 5. Ore de studiu (numar real)
        'hours_studied_weekly': np.random.normal(15, 5, n_samples).round(1),
        
        # 6. Procentaj (numar real)
        'attendance_rate': np.random.normal(85, 10, n_samples).round(1),
        
        # 7. Nota anterioara (numar real)
        'previous_mark': np.random.normal(75, 11, n_samples).round(2),
        
        # 8. Variabila tinta (numar real)
        'exam_score': (np.random.normal(70, 15, n_samples) + 
                      pd.Series(np.random.normal(15, 5, n_samples) * 0.3)
                     ).round(1).clip(0, 100)
    }
    
    # Adauga valori lipsa (5% pe cele 3 coloane numerice)
    for col in ['hours_studied_weekly', 'attendance_rate', 'previous_mark']:
        data[col][np.random.choice(n_samples, int(n_samples * 0.05), replace=False)] = np.nan
    
    df = pd.DataFrame(data)
    df.to_csv('./data/processed/unprepared_data.csv', index=False)
    return df

if __name__ == "__main__":
    generate_student_data()