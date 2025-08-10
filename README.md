# Personality-Prediction-and-Exploratory-Data-Analysis
Personality Prediction and Exploratory Data Analysis
This repository contains a comprehensive Python project that performs exploratory data analysis (EDA), classification modeling, clustering, and dimensionality reduction on a personality dataset. The goal is to predict whether individuals are Introverts or Extroverts based on their behavioral and psychological features.
Project Overview
The project pipeline includes the following key steps:

1. Data Loading and Basic Exploration
Loads a personality dataset (personality_dataset.csv) with various numeric and categorical features.

Prints dataset info and previews data.

Performs statistical summary and frequency counts for categorical columns.

Handles missing values by filling numeric columns with median values and categorical columns with mode values.

2. Exploratory Data Analysis (EDA)
Plots histograms for numeric features to understand their distribution.

Generates a correlation heatmap to visualize relationships among numeric features.

Creates boxplots comparing numeric features by personality type (Introvert vs Extrovert).

Displays personality distribution using both pie charts and bar charts.

3. Data Preprocessing
Encodes categorical variables with LabelEncoder for use in machine learning models.

Splits dataset into training and testing sets (80% train, 20% test).

4. Classification Models
Trains and evaluates three classifiers:

Logistic Regression

Decision Tree

Random Forest

Evaluation metrics reported include Accuracy, Precision, Recall, and F1 Score (weighted average).

Displays detailed classification reports for each model.

Visualizes model performance comparison using bar charts.

Analyzes feature importance from the Random Forest model.

5. Unsupervised Learning — Clustering
Applies PCA to reduce data to two components for visualization.

Performs K-Means clustering to identify natural groupings.

Performs Hierarchical Clustering with Ward linkage and plots a dendrogram.

Compares clustering labels against true personality labels using Adjusted Rand Index (ARI).

Displays crosstabs to inspect cluster-label alignment.

6. Relationship Analysis
Computes and visualizes correlations between:

Stage_fear and Post_frequency

Drained_after_socializing and Time_spent_Alone

Provides scatter plots colored by personality type to explore behavioral links.

7. Dimensionality Reduction Visualizations
Uses PCA and t-SNE to generate 2D visualizations of the dataset, colored by personality class, to visually assess separability of Introverts and Extroverts.
How to Use
Clone the repository and ensure you have all necessary libraries installed:

nginx
Copy
Edit
pip install pandas matplotlib seaborn scikit-learn scipy
Place your dataset file (personality_dataset.csv) in the indicated path or update the file_path variable.

Run the Python script or Jupyter notebook to execute the full pipeline.

Review printed outputs, classification reports, and generated plots for insights.
Dataset
The dataset should contain a target column named "Personality" with values like "Introvert" and "Extrovert".

Includes both numeric and categorical features describing behavioral traits and psychological measures.

Missing values are handled with median/mode imputation.
Results Summary
Model performance metrics provide an understanding of which classification algorithm best predicts personality.

Feature importance highlights which variables most influence the Random Forest model’s predictions.

Clustering analysis reveals how well personality types group naturally without supervision.

Correlation and relationship analysis provides behavioral insights linking specific traits.

Dimensionality reduction plots help visualize the data structure and class separability.
Visualizations Included
Histograms of all numeric features.

Correlation heatmap of numeric variables.

Boxplots comparing features by personality class.

Pie and bar charts for personality distribution.

Model performance comparison bar chart.

Random Forest feature importance bar chart.

PCA and t-SNE scatter plots colored by personality.

K-Means cluster scatter plot (PCA components).

Hierarchical clustering dendrogram.

Scatter plots showing relationships between select features.
License
This project is open source and free to use for educational and research purposes.
