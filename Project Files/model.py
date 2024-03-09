import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split    
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('data.csv', delimiter=";")

# Initialize the ordinal encoder
ordinal_encoder = OrdinalEncoder(categories=[['Graduate', 'Dropout', 'Enrolled']])

# Encode the target variable
df['Target'] = ordinal_encoder.fit_transform(df[['Target']])

# Define features (X) and target variable (y)
y = df['Target']
X = df.drop('Target', axis=1)

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=1)

# Fit the model
rf_model.fit(train_X, train_y)

# Make predictions on the validation set
rf_val_predictions = rf_model.predict(val_X)

# Calculate and print the accuracy score
accuracy = accuracy_score(val_y, rf_val_predictions)
print("Accuracy Score: {:.5f}".format(accuracy))

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: {:.5f}".format(cv_scores.mean()))

# Get feature importances
feature_importances = rf_model.feature_importances_

# Get feature names
feature_names = X.columns

# Create a DataFrame to store feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(20, 10))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=90)  # Rotate all x-axis labels by 90 degrees
plt.show()