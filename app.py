import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

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
    # st.set_page_config(layout="wide", page_title="Titanic Dataset Analysis and Model Training", page_icon=":ship:", initial_sidebar_state="expanded", theme="dark")
    # st.layout("wide")
    # Sidebar - File Upload
    st.sidebar.title('Upload your CSV or Excel file')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    # Display summary message initially
    st.title('Titanic Dataset Analysis and Model Training')
    st.write("""
        Welcome to the Titanic Dataset Analysis and Model Training App.
        This app allows you to upload a Titanic dataset, explore the data, visualize key metrics, 
        and train machine learning models to predict survival.
        Please upload a CSV or Excel file to get started.
    """)

    # If file is uploaded, load the data
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success('File successfully uploaded!')
        except Exception as e:
            st.sidebar.error(f'Error: {e}')
            return

        # Data preparation
        encoded_data = DataPreparation(df)
        encoded_data["Family"] = encoded_data["SibSp"] + encoded_data["Parch"]

        # Drop unnecessary columns
        encoded_data = encoded_data.drop(['PassengerId'], axis=1)

        # Split into X and y
        X = encoded_data.drop("Survived", axis=1)
        y = encoded_data["Survived"]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

        # Model training
        classifiers = {
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier()
        }

        model_performance = {}

        for key, classifier in classifiers.items():
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            cross_val = np.mean(cross_val_score(classifier, X, y, cv=10))
            model_performance[key] = {
                'Accuracy': accuracy,
                'Cross Validation Score': cross_val,
                'Classification Report': classification_report(y_test, predictions, output_dict=True)
            }

        # Remove summary message after file upload
        st.empty()

        # Data Exploration Section
        st.header('Data Exploration')

        # Dataset Shape
        st.subheader('Dataset Shape')
        st.write(f"The dataset has {encoded_data.shape[0]} rows and {encoded_data.shape[1]} columns.")

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

        # Histograms
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

        # Model Performance Comparison
        st.header('Model Performance Comparison')

        # Create dataframe from model_performance dictionary
        df_performance = pd.DataFrame.from_dict({(i, j): model_performance[i][j] 
                                                for i in model_performance.keys() 
                                                for j in model_performance[i].keys()},
                                                orient='index')

        # Plotting model comparison
        st.subheader('Accuracy and Cross Validation Scores')
        st.write(df_performance[['Accuracy', 'Cross Validation Score']])

        fig, ax = plt.subplots(figsize=(10, 8))
        df_performance.plot(kind='bar', ax=ax)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        st.pyplot(fig)

        # Detailed Classification Reports
        st.subheader('Classification Reports')
        for model, report in model_performance.items():
            st.subheader(f'{model} Classification Report')
            st.text_area(f'{model} Classification Report', str(report['Classification Report']))

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
