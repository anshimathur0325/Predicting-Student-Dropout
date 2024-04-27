import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV    
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import sklearn
#removed admission grade
print(sklearn.__version__)
p2 = ['Age at enrollment', 'Course', 'Tuition fees up to date', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Curricular units 1st sem (enrolled)', 'Curricular units 2nd sem (enrolled)']
# Read the data
df = pd.read_csv('data.csv', delimiter=";")
df = df[df['Target'] != 'Enrolled'] 
ins = pd.DataFrame()
# Initialize the ordinal encoder
ordinal_encoder = OrdinalEncoder(categories=[['Graduate', 'Dropout', 'Enrolled']])

# Encode the target variable

df['Target'] = ordinal_encoder.fit_transform(df[['Target']])

# Define features (X) and target variable (y)
y = df['Target']
X = df[p2]
pd.set_option('display.max_columns', None)
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


# Prediction function
def predict(*args):
    global ins
    # Convert user input to DataFrame
    input_data = pd.DataFrame([args], columns=['Age at enrollment', 'Course', 'Tuition fees up to date', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Curricular units 1st sem (enrolled)', 'Curricular units 2nd sem (enrolled)'])
    # Make prediction
    if input_data['Tuition fees up to date'].any():
        input_data['Tuition fees up to date'] = 1
    else:
        input_data['Tuition fees up to date'] = 0

    if (input_data['Course'] == 'Biofuel Production Technologies').any():
        input_data['Course'] = 33
    elif (input_data['Course'] == 'Animation and Multimedia Design').any():
        input_data['Course'] = 171
    elif (input_data['Course'] == 'Social Service').any():
        input_data['Course'] = 9238
    elif (input_data['Course'] == 'Agronomy').any():
        input_data['Course'] = 9003
    elif (input_data['Course'] == 'Communication Design').any():
        input_data['Course'] = 9070
    elif (input_data['Course'] == 'Veterinary Nursing').any():
        input_data['Course'] = 9085
    elif (input_data['Course'] == 'Informatics Engineering').any():
        input_data['Course'] = 9119
    elif (input_data['Course'] == 'Equinculture').any():
        input_data['Course'] = 9130
    elif (input_data['Course'] == 'Management').any():
        input_data['Course'] = 9147
    elif (input_data['Course'] == 'Tourism').any():
        input_data['Course'] = 9254
    elif (input_data['Course'] == 'Nursing').any():
        input_data['Course'] = 9500
    elif (input_data['Course'] == 'Oral Hygiene').any():
        input_data['Course'] = 9556
    elif (input_data['Course'] == 'Advertising and Marketing Management').any():
        input_data['Course'] = 9670
    elif (input_data['Course'] == 'Journalism and Communication').any():
        input_data['Course'] = 9773
    elif (input_data['Course'] == 'Basic Education').any():
        input_data['Course'] = 9853
    elif (input_data['Course'] == 'Management (evening attendance)').any():
        input_data['Course'] = 9991
    elif (input_data['Course'] == 'Social Service (evening attendance)').any():
        input_data['Course'] = 8014

    if (input_data['Curricular units 2nd sem (grade)'] == 'A').any():
        input_data['Curricular units 2nd sem (grade)'] = 20
    elif (input_data['Curricular units 2nd sem (grade)'] == 'B').any():
        input_data['Curricular units 2nd sem (grade)'] = 17
    elif (input_data['Curricular units 2nd sem (grade)'] == 'C').any():
        input_data['Curricular units 2nd sem (grade)'] = 14
    elif (input_data['Curricular units 2nd sem (grade)'] == 'D').any():
        input_data['Curricular units 2nd sem (grade)'] = 12
    elif (input_data['Curricular units 2nd sem (grade)'] == 'E').any():
        input_data['Curricular units 2nd sem (grade)'] = 10
    elif (input_data['Curricular units 2nd sem (grade)'] == 'F').any():
        input_data['Curricular units 2nd sem (grade)'] = 0

    if (input_data['Curricular units 1st sem (grade)'] == 'A').any():
        input_data['Curricular units 1st sem (grade)'] = 19
    elif (input_data['Curricular units 1st sem (grade)'] == 'B').any():
        input_data['Curricular units 1st sem (grade)'] = 16
    elif (input_data['Curricular units 1st sem (grade)'] == 'C').any():
        input_data['Curricular units 1st sem (grade)'] = 14
    elif (input_data['Curricular units 1st sem (grade)'] == 'D').any():
        input_data['Curricular units 1st sem (grade)'] = 12
    elif (input_data['Curricular units 1st sem (grade)'] == 'E').any():
        input_data['Curricular units 1st sem (grade)'] = 10
    elif (input_data['Curricular units 1st sem (grade)'] == 'F').any():
        input_data['Curricular units 1st sem (grade)'] = 0
    prediction = best_rf_model.predict(input_data)
    ins = input_data
    print(ins)
    # Decode prediction
    if (prediction == 0):
        return "Graduate"
    else:
        return "Dropout"


# Create Gradio interface
input_components = [
    gr.Slider(label='Age at enrollment' , minimum=0, maximum=100, step=1),
    gr.Dropdown(label='Course', choices=['Biofuel Production Technologies','Animation and Multimedia Design','Social Service','Agronomy','Communication Design','Veterinary Nursing','Informatics Engineering','Equinculture','Management','Tourism','Nursing','Oral Hygiene','Advertising and Marketing Management','Journalism and Communication','Basic Education', 'Management (evening attendance)',' Social Service (evening attendance)']),
    gr.Checkbox(label='Tuition fees up to date'),
    gr.Slider(label='Approved Classes First Semester', minimum=0, maximum=25, step=1),
    gr.Slider(label='Approved Classes Second Semester', minimum=0, maximum=25, step=1),
    #Grading Scale
    # Less than 10 : FAIL (F)
    # 10: Minimum pass grade (E)
    # 11 - 12: SUFFICIENT (D)
    # 13 - 14: GOOD (C)
    # 15 – 17: VERY GOOD (B)
    # 18 - 20: EXCELLENT (A)
    gr.Dropdown(label='Average Grade First Semester', choices=['A','B','C','D','E','F']),
    gr.Dropdown(label='Average Grade Second Semester', choices=['A','B','C','D','E','F']),
    gr.Slider(label='Enrolled Classes First Semester', minimum=0, maximum=25, step=1),
    gr.Slider(label='Enrolled Classes Second Semester', minimum=0, maximum=25, step=1)
]
def plot_forecast(text):
    global ins
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=text, bins=15, hue='Target', multiple='stack', palette=['skyblue', 'orange'], edgecolor='black', alpha=0.7)
    plt.title('Distribution of '+text)
    plt.xlabel(text)
    plt.ylabel('Frequency')
    plt.grid(False)
    graphl = []
    if (len(ins) != 0):
        print("s")
        user_input = ins[text].iloc[0]
        plt.axvline(x=user_input, color='red', linestyle='--', linewidth=2, label='User Input')
        graphl = [ 'User Input','Dropout', 'Graduate']
    else:
        graphl = [ 'Dropout', 'Graduate']
    
    plt.legend(title='Student Status', labels=graphl)  # Adding labels for the hue
    plt.show()
    return fig

def plot_violin(text):
    df2 = pd.read_csv('./data.csv', delimiter=";")
    fig =plt.figure(figsize=(10,10))
    sns.violinplot(x="Target", y=text, data=df2)
    plt.title(text)
    plt.show()
    return fig

demo = gr.Interface(fn=predict, inputs=input_components, outputs=["text", ], title="Predicting Student Drop Out", description="This model predicts whether a student will drop out or graduate based on the input features. Go to the next page to compare your input values to the students in the dataset.", allow_flagging="never")
# plots = gr.Interface()
inputc = gr.Dropdown(label='Feature', choices=['Age at enrollment', 'Tuition fees up to date', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Curricular units 1st sem (enrolled)', 'Curricular units 2nd sem (enrolled)'])

graphs = gr.Interface(fn=plot_forecast,outputs=gr.Plot(), inputs = inputc, description="Choose a feature and see the overall distribution of students for that specific category. If you used the model in the previous tab, you would be able to see your input for that specific feature.")


with gr.Blocks() as intro:
    gr.set_static_paths(paths=["static/heatmap.png", "static/violin.png"])
    gr.HTML("""
            <h1 style="text-align:center;">Exploratory Data Analysis</h1>
            <div style="font-size:20px">
                <p> This website predits whether someone will drop out or graduate based on various different variables. </p>
                <p> In order to choose features to use in my model, I conducted exploratory data analysis to see the relationships between variables. </p>
            </div>
            <hr style="margin-top:1px;margin-bottom:1px;">
            <div style="font-size:20px">
                <p> The heatmap below shows the relationships between the features of a dataset of students. </p>
                <img src='/file=static/heatmap.png' alt='Heatmap of features and their relationships with each other' height='25%' width='100%' style="margin-bottom:.5em; margin-top:.5em;">
                <p> The features that were chosen are the ones that had the highest correlation with the target variable. The number of credits someone was taking seemed to have the most impact, which was surprising at first. </p>
            
             <p> It was important to me that the input features were easy to understand and interpret. For this reason, I dropped features that a common user of this website would be unable to understand. While this may affect the accuracy of the model, it resulted in a more usable interface. </p>
                <p> This model was trained using a Random Forest Classifier and achieved an accuracy of .90. </p>
                <p> Random Forest Classifer was chosen because it is a powerful and versatile machine learning algorithm that can be used for both classification and regression tasks. It implements multiple decision trees and prevents overfitting the data</p>

            </div>
            """)
inputv = gr.Dropdown(label='Feature', choices=['Age at enrollment', 'Tuition fees up to date', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Curricular units 1st sem (enrolled)', 'Curricular units 2nd sem (enrolled)'])

violins = gr.Interface(fn=plot_violin, outputs=gr.Plot(), inputs = inputv, description="Choose a feature to see its distribution in relation to students who have graduated, dropped out, or still enrolled.")
    

if __name__ == "__main__":
    with gr.Blocks() as x:
        gr.HTML("<h1>Student Drop Out Prediction by Anshi Mathur</h1>")
        gr.TabbedInterface(
        [intro,violins, demo, graphs],
        ["About This Project","EDA viz","Machine Learning Prediction", "Compare Your Inputs to Dataset"]
        )
    x.launch(share=True)