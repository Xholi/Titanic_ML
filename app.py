import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# classifier libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# from pivottablejs import pivot_ui

# Function for label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

def DataPreparation(data):
    for col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])
    return data

# Function to load dataset
@st.cache
def load_data(file):
    df = pd.read_csv(file)
    return df

# Main function to run the Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Titanic Dataset Analysis and Model Training", page_icon=":ship:")

    # Sidebar - File Upload
    st.sidebar.title('Upload your CSV or Excel file')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    # If file is uploaded, load the data
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success('File successfully uploaded!')
        except Exception as e:
            st.sidebar.error(f'Error: {e}')
            return
    else:
        df = pd.read_csv("Titanic_Data.csv")  # Replace with your default dataset path or name
        st.sidebar.success('Default dataset loaded!')

    # Data exploration and preprocessing
    st.title('Titanic Dataset Analysis and Model Training')

    # Display dataset summary
    st.header('Dataset Summary')
    st.subheader('Dataset Shape')
    st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    st.subheader('Data Types')
    st.write(df.dtypes)

    # Pivot Table
    # st.subheader('Pivot Table')
    # pivot_ui(df)

    # Missing Values Heatmap
    st.subheader('Missing Values Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    plt.title('Missing Values')
    st.pyplot(fig)

    # Crosstab and Survival Stats
    st.subheader('Survival Stats by Sex and Pclass (Crosstab)')
    survived = pd.crosstab(
        df['Survived'],
        columns=[df['Sex'], df['Pclass']],
        margins=True
    )
    survived.index = ["Died", "Survived", "Total"]
    df_survival_stats = (survived / survived.loc['Total']) * 100
    st.write(df_survival_stats)

    # Visualizations
    st.header('Visualizations')

    # Countplot Survived
    st.subheader('Countplot of Survived')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Survived', hue='Survived', ax=ax)
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)

    # Histograms by Category
    st.subheader('Histograms by Category')
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    sns.histplot(data=df, x='Pclass', hue='Survived', multiple='stack', ax=axs[0, 0])
    axs[0, 0].set_title('Survived by Pclass')

    sns.histplot(data=df, x='Sex', hue='Survived', multiple='stack', ax=axs[0, 1])
    axs[0, 1].set_title('Survived by Sex')

    sns.histplot(data=df, x='Embarked', hue='Survived', multiple='stack', ax=axs[1, 0])
    axs[1, 0].set_title('Survived by Embarked')

    sns.histplot(data=df, x='Parch', hue='Survived', multiple='stack', ax=axs[1, 1])
    axs[1, 1].set_title('Survived by Parch')

    st.pyplot(fig)

    # Imputing missing values
    st.subheader('Imputing Missing Values')
    st.write("Using random values between median-20 to median+20 for missing values")
    median_age = df['Age'].median()
    lower_limit = median_age - 20
    upper_limit = median_age + 20
    missing_values = df['Age'].isnull().sum()
    random_numbers = np.random.uniform(lower_limit, upper_limit, missing_values)
    df.loc[df['Age'].isnull(), 'Age'] = random_numbers
    st.write("Missing values in 'Age' column imputed.")

    # Encoding categorical variables
    encoded_data = DataPreparation(df)
    encoded_data["Family"] = encoded_data["SibSp"] + encoded_data["Parch"]

    # Drop unnecessary columns
    encoded_data = encoded_data.drop(['PassengerId'], axis=1)

    # Display encoded data
    st.subheader('Encoded Data Sample')
    st.write(encoded_data.head())

    # Correlation Matrix
    st.subheader('Correlation Matrix')
    corr = encoded_data.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Model Training and Evaluation
    st.header('Model Training and Evaluation')

    X = encoded_data.drop("Survived", axis=1)
    y = encoded_data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    classifiers = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }

    for key, classifier in classifiers.items():
        classifier.fit(X_train, y_train)

    # Evaluation report function
    def evaluationReport(model, predictions):
        st.write("============= ", model.__class__.__name__, " Report ==============")
        st.write(f"Accuracy Score : {accuracy_score(y_test, predictions) * 100:.2f}% ")
        st.write(f"Cross Validation Score: {round(np.mean(cross_val_score(model, X, y, cv=10)), 2) * 100:.2f}%")
        st.write("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    # Displaying model performance
    st.subheader('Model Performance Comparison')

    df_performance = pd.DataFrame(columns=['Accuracy', 'Cross Validation Score'])
    for key, classifier in classifiers.items():
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        cv_score = np.mean(cross_val_score(classifier, X, y, cv=10))
        df_performance.loc[key] = [accuracy, cv_score]

    st.write(df_performance)

    fig, ax = plt.subplots(figsize=(10, 8))
    df_performance.plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    st.pyplot(fig)

    # Saving models
    st.header('Saving Models')
    for key, classifier in classifiers.items():
        joblib.dump(classifier, f'XholisileMantshongo_model_{classifier.__class__.__name__}.pkl')
        st.write(f"{classifier.__class__.__name__} model saved.")

    else:
        st.sidebar.info('Awaiting for CSV or Excel file to be uploaded.')

# Run the main function
if __name__ == '__main__':
    main()
