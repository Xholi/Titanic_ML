import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# Function for label encoding
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

# Function to create survival clusters
def create_clusters(data, n_clusters=2):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=3)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters

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
        df = pd.read_csv("Titanic_DAta.csv")  # Replace with your default dataset path or name
        st.sidebar.success('File successfully uploaded!')

    # Data exploration and preprocessing
    st.title('Titanic Dataset Analysis and Model Training')

    # Display dataset summary
    st.header('Dataset Summary')
    st.subheader('Dataset Shape')
    st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    # Missing Values Heatmap
    st.subheader('Missing Values Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    plt.title('Missing Values')
    st.pyplot(fig)

    # Imputing missing values
    st.subheader('Imputing Missing Values')
    st.write("Using random values between median-20 to median+20 for missing values")

    try:
        median_age = df['Age'].median()
        if pd.isnull(median_age):
            median_age = df['Age'].astype(float).median()
    except Exception as e:
        st.sidebar.error(f'Error: {e}')
        return
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
        joblib.dump(classifier, f'model_{classifier.__class__.__name__}.pkl')
        st.write(f"{classifier.__class__.__name__} model saved.")

    # Create and Visualize Survival Clusters
    st.header('Survival Clusters')
    clusters = create_clusters(encoded_data.drop('Survived', axis=1))
    df['Cluster'] = clusters

    st.subheader('Cluster Counts')
    st.write(df['Cluster'].value_counts())

    st.subheader('Clusters Visualization')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='Age', y='Fare', hue='Cluster', palette='viridis', ax=ax)
    plt.title('Clusters Visualization by Age and Fare')
    st.pyplot(fig)

# Run the main function
if __name__ == '__main__':
    main()
