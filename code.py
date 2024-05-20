import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv("diabetes.csv")
df.describe()

df.head()

df.info()

st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

user_data = user_data.rename(columns={
    'age': 'Age',
    'bmi': 'BMI',
    'bp': 'BloodPressure',
    'dpf': 'DiabetesPedigreeFunction',
    'glucose': 'Glucose',
    # ...
})

st.title('Visualised Patient Report')

# Check if there is a function named 'model' defined in the current environment
if 'model' in globals():
    # Print a warning message
    print("Warning: There is a function named 'model' defined in the global environment.")
    print("This may cause unexpected behavior.")

st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['pregnancies'], s = 150, color = 'orange')
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)

# Check if the 'glucose' column exists in the user_data DataFrame
if 'glucose' not in user_data.columns:
    # Add the 'glucose' column to the user_data DataFrame
    user_data['glucose'] = 120  # Replace with the appropriate value

# Check if the 'glucose' column is not empty
if user_data['glucose'].isnull().any():
    # Fill missing values in the 'glucose' column with an appropriate value
    user_data['glucose'].fillna(120, inplace=True)  # Replace with the appropriate value

# Now you can use the 'glucose' column to create the scatterplot
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['glucose'], s = 150, color = 'green')

st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = 'pink')
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)

st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['skinthickness'], s = 150, color = 'blue')
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)

st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['insulin'], s = 150, color = 'blue')
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


if 'bmi' not in user_data.columns:
    # Add the 'bmi' column to the user_data DataFrame
    user_data['bmi'] = 25  # Replace with the appropriate value

# Now you can use the 'bmi' column to create the scatterplot
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['bmi'], s = 150, color = 'green')

try:
    # Try to create the scatterplot using the 'bmi' column
    ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['bmi'], s = 150, color = 'green')
except KeyError:
    # If the 'bmi' column does not exist, print a message
    print("The 'bmi' column does not exist in the user_data DataFrame.")


st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['bmi'], s = 150, color = 'blue')
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

if 'dpf' in user_data.columns:
    # Create the scatterplot using the 'dpf' column
    ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['dpf'], s = 150, color = 'blue')
else:
    # Print a message indicati
    # ng that the 'dpf' column does not exist
    print("The 'dpf' column does not exist in the user_data DataFrame.")


# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Define the user's data
user_data = {
    'Pregnancies': [3],
    'Glucose': [130],
    'BloodPressure': [78],
    'SkinThickness': [0],
    'Insulin': [0],
    'BMI': [33.6],
    'DiabetesPedigreeFunction': [0.627],
    'Age': [47]
}

# Convert user_data to a DataFrame
user_data_df = pd.DataFrame(user_data)

# Train the model
x = df.drop('Outcome', axis=1)
y = df['Outcome']
rf = RandomForestClassifier()
rf.fit(x, y)

# Make predictions for the user's data
user_result = rf.predict(user_data_df)

# Output the prediction result
st.write("Prediction for the user's data: ", "Diabetic" if user_result[0] == 1 else "Not Diabetic")

# Create the BMI scatterplot
fig_bmi = plt.figure()
ax = sns.scatterplot(data=df, x='BMI', y='Age', hue='Outcome')
plt.title('BMI vs Age Scatterplot')
plt.xlabel('BMI')
plt.ylabel('Age')

# Display the scatterplot in Streamlit
st.pyplot(fig_bmi)
