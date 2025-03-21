import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import anderson

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

## 1. Dataset Loading and Missing Value Detection
# Load dataset
df = pd.read_csv("dataset.csv")  # Load the dataset into a pandas DataFrame

# Check for missing values
missing_values = df.isnull().sum()  # Count the number of missing values for each column
print("Missing Values:\n", missing_values)  # Print the number of missing values for each column

# Detect outliers using box plots
numeric_columns = ["Age", "App Sessions", "Distance Travelled (km)", "Calories Burned"]  # Define numeric columns to check for outliers

# Loop through each numeric column and generate box plots to detect outliers
for column in numeric_columns:
    plt.figure(figsize=(6, 4))  # Create a new figure for each box plot
    sns.boxplot(y=df[column])  # Create a box plot for each column
    plt.title(f"Boxplot for {column}")  # Add a title to each box plot
    plt.show()  # Display the box plot

## 2. Categorical Variable Encoding
# Encode categorical variables using ordinal mapping
df_encoded = df.copy()  # Create a copy of the DataFrame to preserve the original dataset
df_encoded["Gender"] = df_encoded["Gender"].map({"Male": 0, "Female": 1})  # Encode Gender (Male=0, Female=1)
df_encoded["Activity Level"] = df_encoded["Activity Level"].map({"Sedentary": 0, "Moderate": 1, "Active": 2})  # Encode Activity Level
df_encoded["Location"] = df_encoded["Location"].map({"Rural": 0, "Suburban": 1, "Urban": 2})  # Encode Location

## 3. Outlier Removal using IQR Method
def remove_outliers(df, column):
    # Convert the column to float64 to avoid dtype conflict
    df[column] = df[column].astype('float64')

    # Calculate Q1 (25th percentile), Q3 (75th percentile), and IQR (Interquartile Range)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outlier detection
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outlier detection

    # Modify values instead of removing rows to handle outliers (capping the outliers)
    df.loc[df[column] < lower_bound, column] = lower_bound  # Cap lower outliers
    df.loc[df[column] > upper_bound, column] = upper_bound  # Cap upper outliers

    return df  # Return the modified DataFrame to preserve structure

# Apply outlier removal to all numerical columns
numeric_columns = ["App Sessions", "Distance Travelled (km)", "Calories Burned", "Age"]  # List of numeric columns to process
for col in numeric_columns:
    df_encoded = remove_outliers(df_encoded, col)  # Apply the outlier removal function to each column

## 4. Distribution Visualization for Numeric Variables
# Visualizing distributions of all numeric variables
plt.figure(figsize=(12, 6))  # Set the figure size for the plot
for column in numeric_columns:
    sns.histplot(df_encoded[column], kde=True, bins=15, label=column, alpha=0.6)  # Plot histogram with KDE (Kernel Density Estimation)
plt.legend()  # Show legend for clarity
plt.title("Distribution of Numeric Variables")  # Add title to the plot
plt.xlabel("Value")  # Label for x-axis
plt.ylabel("Frequency")  # Label for y-axis
plt.show()  # Display the plot

## 5. Individual Variable Distribution Analysis
# Individual variable distributions in separate subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Create subplots (2x2 grid) for individual variable distributions
colors = ["blue", "orange", "green", "red"]  # Set custom colors for each plot

# Loop through each numeric column and create a histogram for each
for i, (column, color) in enumerate(zip(numeric_columns, colors)):
    row, col = i // 2, i % 2  # Calculate row and column for subplot arrangement
    sns.histplot(df_encoded[column], kde=True, bins=15, ax=axes[row, col], color=color)  # Plot each histogram
    axes[row, col].set_title(f"Distribution of {column}")  # Add title to each subplot

plt.tight_layout()  # Adjust spacing between subplots
plt.show()  # Display all subplots

## 6. Correlation Heatmap
# Correlation heatmap to visualize relationships between variables
plt.figure(figsize=(8, 6))  # Set the figure size for the heatmap
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)  # Generate correlation heatmap
plt.title("Correlation Heatmap")  # Title for the heatmap
plt.show()  # Display the heatmap

## 7. Anderson-Darling Normality Test
# Function to perform Anderson-Darling normality test
def normality_test(data, column):
    stat, crit_vals, sig_levels = anderson(data)  # Perform the Anderson-Darling test
    print(f"\nAnderson-Darling Test for {column}:")  # Print the column being tested
    print(f"Test Statistic: {stat:.3f}")  # Print the test statistic
    print("Critical values at different significance levels:")
    for level, crit in zip(sig_levels, crit_vals):
        print(f"  {level:.1f}%: {crit:.3f}")  # Print the critical values at different significance levels

    # Check normality at 5% significance level
    if stat < crit_vals[2]:  # 5% significance level (third critical value)
        print("Conclusion: Fail to reject null hypothesis -> Data appears to be normally distributed.")
    else:
        print("Conclusion: Reject null hypothesis -> Data does not follow a normal distribution.")

# Apply the normality test to each numeric column
numeric_columns = ["Age", "App Sessions", "Distance Travelled (km)", "Calories Burned"]  # List of numeric columns
for col in numeric_columns:
    normality_test(df[col].dropna(), col)  # Perform normality test on each column (dropping any missing values)

## 8. K-Means Clustering
# Selecting features for clustering
X_cluster = df[["App Sessions", "Distance Travelled (km)", "Calories Burned"]]  # Select features for clustering (app sessions, distance, calories)

# Determine the optimal number of clusters using the Elbow method
inertia = []  # List to store inertia values for different k-values
k_values = range(1, 10)  # Define k-values to test (from 1 to 9)

# Loop through k-values and calculate inertia for each
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")  # Apply K-means clustering
    kmeans.fit(X_cluster)  # Fit the K-means model
    inertia.append(kmeans.inertia_)  # Store the inertia value for each k

# Plot the Elbow Method to visualize inertia values for each k
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker="o", linestyle="-")  # Plot inertia values
plt.xlabel("Number of Clusters (k)")  # Label for x-axis
plt.ylabel("Inertia (Within-cluster Sum of Squares)")  # Label for y-axis
plt.title("Elbow Method for Optimal k")  # Title for the plot
plt.show()  # Display the plot

# Apply K-means with the optimal number of clusters (chosen based on elbow method)
optimal_k = 3  # Set the optimal number of clusters to 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")  # Apply K-means with k=3
df["Cluster"] = kmeans.fit_predict(X_cluster)  # Assign cluster labels to the DataFrame

# Cluster Analysis: Compute cluster-wise means for key features
cluster_summary = df.groupby("Cluster")[["App Sessions", "Distance Travelled (km)", "Calories Burned"]].mean()  # Calculate mean values for each cluster
print(cluster_summary)  # Display the cluster summary

## 9. Silhouette Score for Clustering Evaluation
# Compute the Silhouette Score to assess clustering quality
silhouette_avg = silhouette_score(X_cluster, df["Cluster"])  # Calculate Silhouette Score for clustering
print(f"Silhouette Score for k={optimal_k}: {silhouette_avg:.2f}")  # Print the Silhouette Score

## 10. Clustering Visualization
# Visualizing Clusters in 2D
plt.figure(figsize=(8, 6))  # Set figure size for 2D plot
sns.scatterplot(x=df["App Sessions"], y=df["Calories Burned"], hue=df["Cluster"], palette="viridis")  # Plot 2D scatter plot
plt.xlabel("App Sessions")  # Label for x-axis
plt.ylabel("Calories Burned")  # Label for y-axis
plt.title("K-means Clustering Visualization")  # Title for the plot
plt.legend(title="Cluster")  # Add legend to indicate clusters
plt.show()  # Display the plot

# Visualizing Clusters in 3D
fig = plt.figure(figsize=(10, 7))  # Set figure size for 3D plot
ax = fig.add_subplot(111, projection="3d")  # Add 3D subplot
ax.scatter(df["App Sessions"], df["Distance Travelled (km)"], df["Calories Burned"],
           c=df["Cluster"], cmap="viridis", alpha=0.6)  # 3D scatter plot with clusters
ax.set_xlabel("App Sessions")  # Label for x-axis
ax.set_ylabel("Distance Travelled (km)")  # Label for y-axis
ax.set_zlabel("Calories Burned")  # Label for z-axis
ax.set_title("3D Visualization of Clusters")  # Title for the plot
plt.show()  # Display the 3D plot

## 11. Regression Models
# Define the encoded categorical column names
encoded_col_names = ["Gender", "Activity Level", "Location"]

# Simple Linear Regression
X_simple = df_encoded[["App Sessions"]]  # Independent variable (App Sessions)
y = df_encoded["Calories Burned"]  # Dependent variable (Calories Burned)
simple_model = LinearRegression().fit(X_simple, y)  # Fit the model
y_simple_pred = simple_model.predict(X_simple)  # Make predictions

# Multiple Linear Regression
X_multi = df_encoded[["App Sessions", "Distance Travelled (km)"]]  # Independent variables
multi_model = LinearRegression().fit(X_multi, y)  # Fit the model
y_multi_pred = multi_model.predict(X_multi)  # Make predictions

# Extended Multiple Linear Regression (including encoded categorical columns)
X_extended = df_encoded[["App Sessions", "Distance Travelled (km)", "Age"] + encoded_col_names]  # Independent variables
extended_model = LinearRegression().fit(X_extended, y)  # Fit the model
y_extended_pred = extended_model.predict(X_extended)  # Make predictions

# Polynomial Regression (Degree=2)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())  # Polynomial regression pipeline
poly_model.fit(X_extended, y)  # Fit the model
y_poly_pred = poly_model.predict(X_extended)  # Make predictions

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)  # Random Forest model
rf_model.fit(X_extended, y)  # Fit the model
y_rf_pred = rf_model.predict(X_extended)  # Make predictions

## 12. Model Evaluation
# Evaluate all models using MAE, MSE, and R2 score
mae_simple = mean_absolute_error(y, y_simple_pred)  # MAE for Simple Linear Regression
mse_simple = mean_squared_error(y, y_simple_pred)  # MSE for Simple Linear Regression
r2_simple = r2_score(y, y_simple_pred)  # R2 for Simple Linear Regression

mae_multi = mean_absolute_error(y, y_multi_pred)  # MAE for Multiple Linear Regression
mse_multi = mean_squared_error(y, y_multi_pred)  # MSE for Multiple Linear Regression
r2_multi = r2_score(y, y_multi_pred)  # R2 for Multiple Linear Regression

mae_extended = mean_absolute_error(y, y_extended_pred)  # MAE for Extended Multiple Regression
mse_extended = mean_squared_error(y, y_extended_pred)  # MSE for Extended Multiple Regression
r2_extended = r2_score(y, y_extended_pred)  # R2 for Extended Multiple Regression

mae_poly = mean_absolute_error(y, y_poly_pred)  # MAE for Polynomial Regression
mse_poly = mean_squared_error(y, y_poly_pred)  # MSE for Polynomial Regression
r2_poly = r2_score(y, y_poly_pred)  # R2 for Polynomial Regression

mae_rf = mean_absolute_error(y, y_rf_pred)  # MAE for Random Forest Regression
mse_rf = mean_squared_error(y, y_rf_pred)  # MSE for Random Forest Regression
r2_rf = r2_score(y, y_rf_pred)  # R2 for Random Forest Regression

# Store results in a DataFrame
df_performance = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Extended Multiple Regression",
        "Polynomial Regression (Degree=2)",
        "Random Forest Regression"
    ],
    "MAE": [mae_simple, mae_multi, mae_extended, mae_poly, mae_rf],  # MAE for each model
    "MSE": [mse_simple, mse_multi, mse_extended, mse_poly, mse_rf],  # MSE for each model
    "R2": [r2_simple, r2_multi, r2_extended, r2_poly, r2_rf]  # R2 for each model
})

print(df_performance)  # Print the performance results of all models

## 13. Feature Importance in Random Forest Model
importances = rf_model.feature_importances_  # Get feature importance scores from Random Forest
feature_names = X_extended.columns  # Get feature names

# Sort features by importance
sorted_indices = importances.argsort()[::-1]  # Sort features based on importance scores

print("Feature Importance in Random Forest Model:")  # Print feature importance results
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")  # Display feature name and importance score

# Feature names and importance scores (from previous results)
features = ["App Sessions", "Distance Travelled (km)", "Age", "Location", "Gender", "Activity Level"]
importance_scores = [0.8414, 0.0731, 0.0615, 0.0154, 0.0079, 0.0007]

# Create a horizontal bar chart to visualize feature importance
plt.figure(figsize=(8, 5))
plt.barh(features, importance_scores, color='royalblue', edgecolor='black')  # Create horizontal bar chart
plt.xlabel("Importance Score")  # Label for x-axis
plt.ylabel("Feature")  # Label for y-axis
plt.title("Feature Importance in Random Forest Model")  # Title for the plot
plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top

# Display the plot
plt.show()

## 14. Predicted vs Actual and Residual Plot
# Predicted vs Actual Calories Burned for Random Forest Regression
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_rf_pred, alpha=0.6)  # Scatter plot comparing predicted vs actual values
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--")  # Add a reference line (perfect prediction)
plt.xlabel("Actual Calories Burned")  # Label for x-axis
plt.ylabel("Predicted Calories Burned")  # Label for y-axis
plt.title("Predicted vs. Actual Calories Burned (Random Forest)")  # Title for the plot
plt.show()

# Residual Plot for Random Forest Regression
plt.figure(figsize=(8, 6))
sns.residplot(x=y_rf_pred, y=(y_rf_pred - y), lowess=True, color="blue")  # Residual plot
plt.xlabel("Predicted Calories Burned")  # Label for x-axis
plt.ylabel("Residuals")  # Label for y-axis
plt.title("Residual Plot for Random Forest Regression")  # Title for the plot
plt.show()
