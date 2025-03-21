import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define column names
column_names = ["sepallength", "sepalwidth", "petallength", "petalwidth", "class"]

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None, names=column_names)

# Check the first few rows to ensure the data is loaded correctly
print("Top 5 rows of the dataset:")
print(df.head())  # Prints the first 5 rows

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())  # Check for missing values

# Initial data cleansing
# Check the data types
print("\nData types of each column:")
print(df.dtypes)

# Ensure that the 'class' column is categorical
df['class'] = df['class'].astype('category')

# Handle missing values:
# Impute missing numerical columns with mean (only for numerical columns)
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Impute missing categorical 'class' column with mode
df['class'] = df['class'].fillna(df['class'].mode()[0])

# Calculate Correlations Between Features (excluding the 'class' column)
correlation_matrix = df.iloc[:, :-1].corr()  # Exclude the 'class' column
print("\nCorrelation matrix:")
print(correlation_matrix)

# Visualize the Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Iris Dataset Features")
plt.show()
