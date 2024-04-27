import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV    
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
p2 = ['Curricular units 2nd sem (approved)','Curricular units 1st sem (approved)','Curricular units 2nd sem (grade)','Tuition fees up to date','Curricular units 1st sem (grade)','Age at enrollment','Admission grade','Course','Previous qualification (grade)','Curricular units 2nd sem (evaluations)','Curricular units 1st sem (evaluations)','Curricular units 2nd sem (enrolled)','Curricular units 1st sem (enrolled)']
# Read the data
df = pd.read_csv('data.csv', delimiter=";")
df = df[df['Target'] != 'Enrolled'] 
# Initialize the ordinal encoder
ordinal_encoder = OrdinalEncoder(categories=[['Graduate', 'Dropout', 'Enrolled']])

# Encode the target variable
df['Target'] = ordinal_encoder.fit_transform(df[['Target']])
# Define features (X) and target variable (y)
y = df['Target']
X = df[p2]

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=1)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the model
grid_search.fit(train_X, train_y)
print("Random Forest")
print("")
# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Perform cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: {:.5f}".format(cv_scores.mean()))

# Make predictions on the validation set using the best model
best_rf_model = grid_search.best_estimator_
rf_val_predictions = best_rf_model.predict(val_X)

# Calculate and print the accuracy score
accuracy = accuracy_score(val_y, rf_val_predictions)
print("Validation Accuracy Score: {:.5f}".format(accuracy))

# Calculate and print the F1 score
f1 = f1_score(val_y, rf_val_predictions, average='weighted')
print("Validation F1 Score: {:.5f}".format(f1))