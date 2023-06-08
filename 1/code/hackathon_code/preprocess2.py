from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

# Define the steps in the pipeline
steps = [
    ('encoder',  OneHotEncoder(min_frequency=0.2)),
    ('scaler', StandardScaler()),               # Step 1: Data scaling
    ('feature_selection', SelectKBest(k=5)),    # Step 2: Feature selection
    ('classifier', LogisticRegression())        # Step 3: Classifier
]

# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = pipeline.predict(X_test)