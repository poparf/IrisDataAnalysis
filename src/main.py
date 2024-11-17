# Here we will prepare the iris dataset for PCA and clustering...
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data # this contains the 50 samples of each of the 3 species of iris flowers, so 150 samples in total.
print("Number of records:" + str(len(X)))

y = iris.target
print("Target values represents the label for each flower in the dataset. The target values are:")
print(
"""
0 = Iris Setosa
1 = Iris Versicolor
2 = Iris Virginica
""")
feature_names = iris.feature_names

# Create a DataFrame for easier handling
# A pandas dataframe is a two-dimensional, tabular data structure with labeled axes (rows and columns).
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['target_names'] = df['target'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})
"""
This is how the dataframe looks like:

     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target target_names
0                  5.1               3.5                1.4               0.2       0       setosa
1                  4.9               3.0                1.4               0.2       0       setosa
2                  4.7               3.2                1.3               0.2       0       setosa

"""

# It is really important to standaridze the features for a correct PCA analysis.
# It involves rescaling each feature such that it has a standard deviation of 1 and a mean of 0.
# Dimensional reduction using PCA consists of finding the features that maximize the variance.
# If one feature varies more than the others only because of their respective scales, PCA would determine that such feature dominates the direction of the principal components.
# Why standardize? Because PCA is a variance maximizing exercise, and variables with large variances will dominate over those with small variances.
# Standardizing the features will ensure that each feature contributes equally to the analysis.
# Imagine comparing atheletes performances. You wouldn't compare the number of goals in Handball with the number of goals in Football without converting
# first to a common scale, like percentage of goals scored in the game.
X_scaled = StandardScaler().fit_transform(X)

