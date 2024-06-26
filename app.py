import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix)
import joblib
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

label_encoder = LabelEncoder()

def DataPreparation(data):
    for col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])
    return data

def create_and_evaluate_models(X_train, y_train, X_test, y_test):
    classifiers = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }

    results = {}
    for key, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train, cv=16)
        results[key] = {
            'training_score': round(training_score.mean(), 2) * 100,
            'model': classifier
        }
    return results

def grid_search_tuning(X_train, y_train):
    # Logistic Regression
    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
    grid_log_reg.fit(X_train, y_train)
    log_reg = grid_log_reg.best_estimator_

    # RandomForest Classifier
    RandForest_params = {'max_features': list(range(2, 5, 1)), "max_depth": list(range(2, 4, 1)), 'ccp_alpha': [0.001, 0.01, 0.02, 0.1, 0.2], 'criterion': list({"gini", "entropy", "log_loss"})}
    grid_randforest = GridSearchCV(RandomForestClassifier(), RandForest_params)
    grid_randforest.fit(X_train, y_train)
    random_forest = grid_randforest.best_estimator_

    # DecisionTree Classifier
    tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)), "min_samples_leaf": list(range(5, 7, 1))}
    grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
    grid_tree.fit(X_train, y_train)
    tree_clf = grid_tree.best_estimator_

    return log_reg, random_forest, tree_clf

def evaluation_report(model, predictions, X_test, y_test):
    return {
        "accuracy_score": model.score(X_test, y_test) * 100,
        "cross_val_score": round(np.mean(cross_val_score(model, X_test, y_test, cv=10)), 2) * 100,
        "confusion_matrix": confusion_matrix(y_test, predictions)
    }

def main():
    st.set_page_config(layout="wide", page_title="Titanic Dataset Analysis and Model Training", page_icon=":ship:")

    st.sidebar.title('Upload your CSV file')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('Titanic_DAta.csv')
    
    st.title('Titanic Dataset Analysis and Model Training')

    st.header('Dataset Summary')
    st.subheader('Dataset Shape')
    st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    st.subheader('Dataset Head')
    st.write(df.head())

    women = df.loc[df.Sex == 'female']["Survived"]
    rate_women = sum(women)/len(women)
    st.write("% of women who survived:", round(rate_women * 100, 2))

    men = df.loc[df.Sex == 'male']["Survived"]
    rate_men = sum(men)/len(men)
    st.write("% of men who survived:", round(rate_men * 100, 2))

    st.subheader('Survival Crosstab')
    survived = pd.crosstab(df['Survived'], columns=[df['Sex'], df['Pclass']], margins=True)
    survived.index = ["Die", "Survive", "Total"]
    df_survival_stats = (survived / survived.loc['Total']) * 100
    st.write(df_survival_stats)

    st.write("From the crosstab we can conclude that class has a direct impact on survival rate as well as gender:")
    st.write("1. It is safe to conclude that the lower the class the more likely to die, class order goes as follows: 1 -Being the highest and 3-Being the lowest.")
    st.write("2. More men died, compared to females. This is also in tandem with class as well.")
    st.write(f"3. In Percentage terms, {round(rate_men * 100, 2)}% of men survived compared to {round(rate_women * 100, 2)}% survival rate for women.")

    st.subheader('Missing Values Heatmap')
    fig = px.imshow(df.isnull(), color_continuous_scale='Viridis', aspect='auto')
    st.plotly_chart(fig)

    st.subheader('Data Imputation')
    median = 28
    lower_limit = median - 20
    upper_limit = median + 20
    missing_values = df['Age'].isnull().sum()
    random_numbers = np.random.uniform(lower_limit, upper_limit, missing_values)
    df["Age"] = df["Age"].fillna(pd.Series(random_numbers, index=df.index[df["Age"].isnull()]))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0])
    st.write("Missing values filled.")

    st.subheader('Pairplot')
    fig = sns.pairplot(df, markers='o')
    st.pyplot(fig)

    st.subheader('Age Distribution')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    fig = px.histogram(df, x='Age', color='Survived', nbins=10, marginal='box', title="Age Distribution")
    st.plotly_chart(fig)

    encoded_data = DataPreparation(df)

    st.subheader('Correlation Matrix')
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(encoded_data.corr(), cmap='cool', annot=True, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    encoded_data = encoded_data.drop(['PassengerId'], axis=1)
    encoded_data["Family"] = encoded_data["SibSp"] + encoded_data["Parch"]

    X = encoded_data.drop("Survived", axis=1)
    y = encoded_data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    st.header('Model Training and Evaluation')

    st.subheader('Training Models')
    models = create_and_evaluate_models(X_train, y_train, X_test, y_test)
    for key, result in models.items():
        st.write(f"Classifiers: {key} has a training score of {result['training_score']}% accuracy score")
    
    log_reg, random_forest, tree_clf = grid_search_tuning(X_train, y_train)

    st.subheader('Cross Validation Scores')
    log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=10)
    st.write('Logistic Regression Cross Validation Score:', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
    rf_score = cross_val_score(random_forest, X_train, y_train, cv=10)
    st.write('Random Forest Cross Validation Score:', round(rf_score.mean() * 100, 2).astype(str) + '%')
    tree_score = cross_val_score(tree_clf, X_train, y_train, cv=10)
    st.write('DecisionTree Classifier Cross Validation Score:', round(tree_score.mean() * 100, 2).astype(str) + '%')

    st.subheader('Validation Accuracy Scores')
    rf_val_predictions = random_forest.predict(X_test)
    st.write("Accuracy Random Forest Classifier:", accuracy_score(y_test, rf_val_predictions) * 100)
    lr_val_predictions = log_reg.predict(X_test)
    st.write("Accuracy Logistic Regression:", accuracy_score(y_test, lr_val_predictions) * 100)
    dt_val_predictions = tree_clf.predict(X_test)
    st.write("Accuracy Decision Tree Classifier:", accuracy_score(y_test, dt_val_predictions) * 100)

    st.subheader('Evaluation Reports')
    evaluation_reports = {
        "Logistic Regression": evaluation_report(log_reg, lr_val_predictions, X_test, y_test),
        "Random Forest": evaluation_report(random_forest, rf_val_predictions, X_test, y_test),
        "Decision Tree": evaluation_report(tree_clf, dt_val_predictions, X_test, y_test)
    }

    # st.write(evaluation_reports)

    # Bar graph to compare evaluation scores
    scores = {
        'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree'],
        'Accuracy': [evaluation_reports['Logistic Regression']['accuracy_score'], 
                     evaluation_reports['Random Forest']['accuracy_score'], 
                     evaluation_reports['Decision Tree']['accuracy_score']],
        'Cross Validation': [evaluation_reports['Logistic Regression']['cross_val_score'],
                             evaluation_reports['Random Forest']['cross_val_score'],
                             evaluation_reports['Decision Tree']['cross_val_score']]
    }
    
    scores_df = pd.DataFrame(scores)
    fig = px.bar(scores_df, x='Model', y=['Accuracy', 'Cross Validation'], barmode='group', title="Model Comparison")
    st.plotly_chart(fig)

    # Additional visuals for insights
    st.subheader('Fare Distribution')
    fig = px.histogram(df, x='Fare', color='Survived', nbins=10, marginal='box', title="Fare Distribution")
    st.plotly_chart(fig)

    st.subheader('3D Scatter Plot of Age, Fare, and Survival')
    fig = px.scatter_3d(df, x='Age', y='Fare', z='Survived', color='Survived', symbol='Sex', 
                        title="3D Scatter Plot of Age, Fare, and Survival", height=800)
    st.plotly_chart(fig)

    st.subheader('PCA 2D Scatter Plot')
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(encoded_data)
    fig = px.scatter(pca_components, x=0, y=1, color=df['Survived'], title="PCA 2D Scatter Plot", height=800)
    st.plotly_chart(fig)

    st.subheader('t-SNE 2D Scatter Plot')
    tsne = TSNE(n_components=2)
    tsne_components = tsne.fit_transform(encoded_data)
    fig = px.scatter(tsne_components, x=0, y=1, color=df['Survived'], title="t-SNE 2D Scatter Plot", height=800)
    st.plotly_chart(fig)

    joblib.dump(log_reg, 'logistic_regression_model.pkl')
    joblib.dump(random_forest, 'random_forest_model.pkl')
    joblib.dump(tree_clf, 'decision_tree_model.pkl')
    st.write("Models saved as logistic_regression_model.pkl, random_forest_model.pkl, and decision_tree_model.pkl")

if __name__ == '__main__':
    main()
