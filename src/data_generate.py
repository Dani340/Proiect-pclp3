# CARSTOIU Daniel-Petru 313CA
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
			["Computer Science", "Mathematics", "Psychology", "Electrical Engineering", "Nuclear Sciences"], 
			n_samples,
			p=[0.25, 0.25, 0.2, 0.2, 0.1]
		),
		
		# 3. Varsta (numar intreg)
		'age': np.random.randint(18, 25, n_samples),
		
		# 4. Bursa (val true/false)
		'has_scholarship': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
		
		# 5. Ore de studiu (numar real)
		'hours_studied_weekly': np.random.normal(10, 5, n_samples).round(1),
		
		# 6. Prezenta (numar real)
		'attendance_rate': np.random.normal(70, 10, n_samples).round(1),
		
		# 7. Nota anterioara (numar real)
		'previous_mark': np.random.normal(7.5, 1.1, n_samples).round(2),
	}

	# 8. Variabila tinta (numar real)
	score = (
		data['previous_mark'] * 0.7 +
		data['hours_studied_weekly'] * 0.15 +
		data['attendance_rate'] * 0.02 +
		np.where(data['has_scholarship'], 0.3, 0)
	)
	data['exam_score'] = score.clip(0, 10).round(2)
	
	# Adauga valori lipsa (5% pe cele 3 coloane numerice)
	for col in ['hours_studied_weekly', 'attendance_rate', 'previous_mark']:
		data[col][np.random.choice(n_samples, int(n_samples * 0.05), replace=False)] = np.nan
	
	df = pd.DataFrame(data)
	df.to_csv('./data/unprepared_data.csv', index=False)
	return df

if __name__ == "__main__":
	generate_student_data()