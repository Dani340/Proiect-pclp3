# CARSTOIU Daniel-Petru 313CA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.figsize"] = (10, 6)

def load_data():
	train = pd.read_csv('./data/train.csv')
	test = pd.read_csv('./data/test.csv')
	return train, test

def describe_data(df, name):
	print(f"\nDescriptive statistics for {name}")
	print(df.describe(include='all'))

def plot_distributions(df, name):
	print(f"\nDistribution plots for {name} generated")

	numeric_cols = df.select_dtypes(include=[np.number]).columns
	categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

	# Histograme pentru variabile numerice
	for col in numeric_cols:
		plt.figure()
		sns.histplot(df[col], kde=True, bins=30)
		plt.title(f"{name} - Histograma: {col}")
		plt.xlabel(col)
		plt.ylabel("Frecventa")
		plt.tight_layout()
		plt.savefig(f"./docs/plots/histogram_{col}.png", bbox_inches='tight')
		plt.close()

	# Countplot pentru variabile categorice
	for col in categorical_cols:
		plt.figure()
		sns.countplot(x=col, data=df)
		plt.title(f"{name} - Countplot: {col}")
		plt.xticks(rotation=30)
		plt.tight_layout()
		plt.savefig(f"./docs/plots/countplot_{col}.png", bbox_inches='tight')
		plt.close()

def detect_outliers(df, name):
	print(f"\nOutlier detection for {name}")
	numeric_cols = df.select_dtypes(include=[np.number]).columns

	for col in numeric_cols:
		first = df[col].quantile(0.25)
		third = df[col].quantile(0.75)
		IQR = third - first

		lower_bound = first - 1.5 * IQR
		upper_bound = third + 1.5 * IQR
		outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
		print(f"{col}: {len(outliers)} outliers")

		plt.figure()
		sns.boxplot(x=df[col])
		plt.title(f"{name} - Boxplot: {col}")
		plt.tight_layout()
		plt.savefig(f"./docs/plots/boxplot_{col}.png", bbox_inches='tight')
		plt.close()

def correlation_analysis(df):
	print("\nCorrelation matrix generated")
	numeric_cols = df.select_dtypes(include=[np.number])
	corr = numeric_cols.corr()

	plt.figure(figsize=(10, 8))
	sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
	plt.title("Heatmap - Corelatii intre variabile numerice")
	plt.tight_layout()
	plt.savefig("./docs/plots/heatmap_correlations.png", bbox_inches='tight')
	plt.close()

def relationship_with_target(df):
	print("\nRelationship with target (exam_score) generated")
	numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("exam_score")

	for col in numeric_cols:
		plt.figure()
		sns.scatterplot(x=col, y="exam_score", data=df)
		plt.title(f"Scatter plot: {col} vs. exam_score")
		plt.tight_layout()
		plt.savefig(f"./docs/plots/scatter_{col}_vs_exam_score.png", bbox_inches='tight')
		plt.close()

	categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
	for col in categorical_cols:
		plt.figure()
		sns.violinplot(x=col, y="exam_score", data=df)
		plt.title(f"Violin plot: exam_score by {col}")
		plt.xticks(rotation=30)
		plt.tight_layout()
		plt.savefig(f"./docs/plots/violin_exam_score_by_{col}.png", bbox_inches='tight')
		plt.close()

def train_and_evaluate(train, test):
	print("\nModel: Regresie Liniara")
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.compose import ColumnTransformer
	from sklearn.pipeline import Pipeline

	numeric_features = ['age', 'hours_studied_weekly', 'attendance_rate', 'previous_mark']
	categorical_features = ['gender', 'faculty', 'has_scholarship']

	# Separare X și y
	X_train = train[numeric_features + categorical_features]
	y_train = train['exam_score']
	X_test = test[numeric_features + categorical_features]
	y_test = test['exam_score']

	# One-hot encoding pentru var categorice
	preprocessor = ColumnTransformer([
		('cat', OneHotEncoder(drop='first'), categorical_features)
	], remainder='passthrough')

	model = Pipeline([
		('preprocessing', preprocessor),
		('regressor', LinearRegression())
	])

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	# Metrici
	print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
	print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
	print("R^2:", round(r2_score(y_test, y_pred), 3))

	# Plot eroare
	plt.figure()
	sns.scatterplot(x=y_test, y=y_pred)
	plt.plot([0, 100], [0, 100], '--', color='red')
	plt.xlabel("Valori reale")
	plt.ylabel("Predictii")
	plt.title("Predictii vs. Valori reale")
	plt.tight_layout()
	plt.savefig("./docs/plots/predictions_vs_actual.png", bbox_inches='tight')
	plt.close()

if __name__ == "__main__":
	train_df, test_df = load_data()
	describe_data(train_df, "Train set")
	describe_data(test_df, "Test set")
	plot_distributions(train_df, "Train set")
	plot_distributions(test_df, "Test set")
	detect_outliers(train_df, "Train set")
	correlation_analysis(train_df)
	relationship_with_target(train_df)
	train_and_evaluate(train_df, test_df)
