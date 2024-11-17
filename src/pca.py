import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# A library for making statistical graphics in python. It is built on top of matplotlib and closely integrated with pandas data structures.
import seaborn as sns

from main import df, feature_names, X, y, X_scaled

# PART 1 - CORRELATION MATRIX

# First, let's look at the correlations between original features
plt.figure(figsize=(10, 8)) # width and height in inches

#  - df[feature_names] selects the columns of the dataframe that are the features. corr() calculates the correlation between the features.
#  - annot=True adds the correlation values to the heatmap on each cell.
#  - cmap='coolwarm' sets the color palette.
sns.heatmap(df[feature_names].corr(), annot=True, cmap='coolwarm')

plt.title('Correlation Matrix of Original Features')
# tight_layout automatically adjusts the layout.
plt.tight_layout()
plt.savefig('../output/pca/correlation_matrix.png')
plt.close()

# INTERPRETATION
# The correlation matrix shows that the petal length and petal width are highly correlated with each other, and also highly correlated with sepal length.
# Sepal width has the lowest correlation with the other features.


# 3. Apply PCA
# PCA is a technique for reducing the dimensionality of such datasets, increasing interpretability but at the same time minimizing information loss.
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 4. Calculate explained variance ratio
# This returns the variance caused by each of the principal components.
# The explained variance tells you how much information (variance) can be attributed to each of the principal components.
# Variance is a measure of the variability or spread in the data points.
# The cumulative explained variance tells you how much of the total variance is contained within the first N components.
explained_variance_ratio = pca.explained_variance_ratio_  # [0.72962445 0.22850762 0.03668922 0.00517871]
print("Explained variance ratio:",explained_variance_ratio)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio) # [0.72962445 0.95813207 0.99482129 1.]
# Cumulative means the following:
# PC1: 72%
# PC1 + PC2: 95%
# PC1 + PC2 + PC3: 99%
# PC1 + PC2 + PC3 + PC4: 100%
print("Cumulative explained variance ratio:",cumulative_variance_ratio)

# Plot explained variance
plt.figure(figsize=(10, 6))
# The X-axis shows the number of components ( 1 to 4 )
# Y-axis shows the cumulative explained variance ratio
plt.plot(range(1, len(explained_variance_ratio) + 1), 
         cumulative_variance_ratio, 'bo-') # bo- is a blue line with circles
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs Number of Components')
plt.axhline(y=0.95, color='r', linestyle='--', 
           label='95% Explained Variance')
plt.legend()
plt.grid(True)
plt.savefig('../output/pca/explained_variance.png')
plt.close()

# How to use this plot ?
# If you see that 2 components explain 95% of variance, you can reduce your data from 4 dimensions to 2 dimension while keeping 95% of information !
# This is HUGE ! It means that you can reduce the complexity of your data by 50% while keeping 95% of the information !

# 5. Visualize the first two principal components
plt.figure(figsize=(12, 8))

# We select only the first and the second component.
# Remember !
# PCs are not direct columns, but combinations of original features
# Each PC is a weighted combination of the original features.

# Example:
# PC1 might be something like:
# 0.5 × (sepal length) + 0.5 × (petal length) + 0.4 × (petal width) - 0.2 × (sepal width)
# PC2 might be:
# -0.3 × (sepal length) + 0.8 × (sepal width) - 0.2 × (petal length) - 0.1 × (petal width)

# c=y assigns a color to each species of iris flower
# cmap='viridis' is the color map
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                     cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Iris Dataset - First Two Principal Components')
# Create a legend with custom labels
handles, _ = scatter.legend_elements()
labels = ['Setosa', 'Versicolor', 'Virginica']
plt.legend(handles, labels)
plt.savefig('../output/pca/pca_scatter.png')
plt.close()

# As you can see we've reduced 4 dimensions to 2 dimensions.
# Each point in the scatter plot represents a iris flower.
# Each color represents a different species.
# Purple is Setosa, Yellow is Virginica, Green is Versicolor.
# Setoa is completely separated from the other two species.
# Versicolor and Virginica show some overlap.
# Setosa is very different from the other two species.
# Remember: The distance between points shows how similar or different the flowers are.

# Loadings are the weights of the original features in the principal components.
# Range from -1 to 1
# The higher the absolute value, the more important the feature is for that component.
# Red - positive correlation
# Blue - negative correlation
# White/Light - weak correlation
# It helps you understand what each PC means in terms of the original features.
# 6. Create a loadings plot
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2', 'PC3', 'PC4'],
    index=feature_names
)

plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title('PCA Loadings')
plt.savefig('../output/pca/pca_loadings.png')
plt.close()

# PC1 represents overall the petal length, width, sepal length almost equally
# but sepal width is not important.
# PC2 is heavely influenced by sepal width and less of sepal length.
# PC3 is influenced by sepal length and negatively influenced by petal width. 
# PC 4 is postively influenced by petal length.

print("\nExplained Variance Ratio:", explained_variance_ratio)
print("\nCumulative Explained Variance Ratio:", cumulative_variance_ratio)
print("\nComponent Loadings:\n", loadings)

with open('../output/pca/pca_analysis_results.txt', 'w') as f:
    f.write("PCA Analysis Results for Iris Dataset\n")
    f.write("=====================================\n\n")
    
    f.write("Explained Variance Ratio:\n")
    for i, var in enumerate(explained_variance_ratio, 1):
        f.write(f"PC{i}: {var:.4f}\n")
    
    f.write("\nCumulative Explained Variance Ratio:\n")
    for i, var in enumerate(cumulative_variance_ratio, 1):
        f.write(f"PC{i}: {var:.4f}\n")
    
    f.write("\nComponent Loadings:\n")
    f.write(loadings.to_string())