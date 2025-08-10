import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------- Load Dataset --------------------
file_path = r"C:\Users\Muttaki\Downloads\archive\personality_dataset.csv"
df = pd.read_csv(file_path)

# -------------------- Basic Info --------------------
print("Dataset Info:")
print(df.info())
print("\nDataset Head:")
print(df.head())

# -------------------- Basic EDA --------------------
stats_numeric = df.describe()
stats_categorical = {col: df[col].value_counts(dropna=False)
                     for col in df.select_dtypes(include='object').columns}

print("\nCategorical (count form)")
for col in df.select_dtypes(include='object').columns:
    print(f"\nFrequency counts for {col}:")
    print(df[col].value_counts(dropna=False))

print("\nCategorical (percentage form)")
for col in df.select_dtypes(include='object').columns:
    print(f"\nFrequency counts (%) for {col}:")
    print(df[col].value_counts(normalize=True, dropna=False) * 100)

# -------------------- Missing Value Treatment --------------------
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print("\nMissing values after treatment:")
print(df.isnull().sum())
print("\nMissing values percentage per column:")
print((df.isnull().sum() / len(df)) * 100)

# -------------------- Histograms --------------------
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot((len(num_cols) // 3) + 1, 3, i)
    df[col].hist(bins=20, color='maroon', edgecolor='red')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# -------------------- Heatmap --------------------
print('Heatmap')
corr_matrix = df[num_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features", fontsize=14)
plt.show()

# -------------------- Boxplots --------------------
numeric_features = [col for col in num_cols if col != 'Personality']
fig, axes = plt.subplots(nrows=1, ncols=len(numeric_features), figsize=(5 * len(numeric_features), 6))
if len(numeric_features) == 1:
    axes = [axes]
for ax, feature in zip(axes, numeric_features):
    sns.boxplot(data=df, x='Personality', y=feature, ax=ax)
    ax.set_title(f"Boxplot of {feature} by Personality")
plt.tight_layout()
plt.show()

# -------------------- Pie chart --------------------
personality_counts = df['Personality'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(personality_counts, labels=personality_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Personality Distribution')
plt.show()

# -------------------- Bar chart --------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Personality', palette='pastel')
plt.title('Personality Distribution')
plt.ylabel('Count')
plt.xlabel('Personality Type')
plt.show()

# -------------------- Encode categorical variables --------------------
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------- Prepare features and target --------------------
X = df.drop('Personality', axis=1)
y = df['Personality']

# -------------------- Train-test split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- Model training --------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    }

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    result = evaluate_model(name, model, X_test, y_test)
    results.append(result)

results_df = pd.DataFrame(results)
print("\nSummary of Model Performance:")
print(results_df)

plt.figure(figsize=(8, 5))
sns.barplot(data=results_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Metric', y='Score', hue='Model')
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.show()

# -------------------- Feature importance from Random Forest --------------------
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
features = X.columns

feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importance Ranking (Random Forest)')
plt.tight_layout()
plt.show()

# -------------------- Clustering Analysis --------------------

# PCA to reduce to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=len(y.unique()), random_state=42)
clusters_km = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters_km, palette='Set2', s=50, legend='full')
plt.title('K-Means Clusters Visualized on PCA Components')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Hierarchical Clustering (Ward linkage)
Z = linkage(X, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('Sample index or cluster size')
plt.ylabel('Distance')
plt.show()

# Get cluster assignments from hierarchical clustering
clusters_hc = fcluster(Z, t=len(y.unique()), criterion='maxclust')

# -------------------- Cluster vs Label Alignment --------------------
ari_km = adjusted_rand_score(y, clusters_km)
ari_hc = adjusted_rand_score(y, clusters_hc)

print(f"\nAdjusted Rand Index (K-Means vs True labels): {ari_km:.4f}")
print(f"Adjusted Rand Index (Hierarchical Clustering vs True labels): {ari_hc:.4f}")

print("\nCrosstab: K-Means clusters vs Personality labels")
print(pd.crosstab(clusters_km, y))

print("\nCrosstab: Hierarchical clusters vs Personality labels")
print(pd.crosstab(clusters_hc, y))

# -------------------- 6. Relationship Analysis --------------------

# Correlation between Stage_fear and Post_frequency
if 'Stage_fear' in df.columns and 'Post_frequency' in df.columns:
    corr_sf_pf = df['Stage_fear'].corr(df['Post_frequency'])
    print(f"\nCorrelation between Stage_fear and Post_frequency: {corr_sf_pf:.4f}")
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x='Stage_fear', y='Post_frequency', hue='Personality', palette='tab10')
    plt.title('Stage_fear vs Post_frequency')
    plt.show()
else:
    print("\nColumns 'Stage_fear' and/or 'Post_frequency' not found in dataset.")

# Relationship between Drained_after_socializing and Time_spent_Alone
if 'Drained_after_socializing' in df.columns and 'Time_spent_Alone' in df.columns:
    corr_ds_ta = df['Drained_after_socializing'].corr(df['Time_spent_Alone'])
    print(f"\nCorrelation between Drained_after_socializing and Time_spent_Alone: {corr_ds_ta:.4f}")
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x='Drained_after_socializing', y='Time_spent_Alone', hue='Personality', palette='tab10')
    plt.title('Drained_after_socializing vs Time_spent_Alone')
    plt.show()
else:
    print("\nColumns 'Drained_after_socializing' and/or 'Time_spent_Alone' not found in dataset.")

# -------------------- 7. Dimensionality Reduction --------------------

# PCA 2D visualization
pca_2 = PCA(n_components=2, random_state=42)
X_pca2 = pca_2.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca2[:,0], y=X_pca2[:,1], hue=y, palette='Set1', legend='full', s=60)
plt.title('PCA 2D Visualization Colored by Personality')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Personality')
plt.show()

# t-SNE 2D visualization (can be slow)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='Set1', legend='full', s=60)
plt.title('t-SNE 2D Visualization Colored by Personality')
plt.xlabel('t-SNE Dim 1')
plt.ylabel('t-SNE Dim 2')
plt.legend(title='Personality')
plt.show()
