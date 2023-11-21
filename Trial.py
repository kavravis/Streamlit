import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DataHandler:
    def upload_csv(self):
        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("File uploaded successfully!")
                return df
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        return None

    def display_uploaded_data(self, df):
        st.header("Uploaded CSV Data:")
        st.write(df)

    def clean_data(self, df, clean_option):
        df_cleaned = None
        if clean_option != "None":
            if clean_option == "Drop Missing Values":
                df_cleaned = df.dropna()
            elif clean_option == "Impute with Mean":
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df_cleaned = df.copy()
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())
            elif clean_option == "Impute with Median":
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df_cleaned = df.copy()
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
            elif clean_option == "Impute with Mode":
                df_cleaned = df.fillna(df.mode().iloc[0])
        return df_cleaned

    def display_cleaned_data(self, df_cleaned):
        st.header("Cleaned Data:")
        st.write(df_cleaned)

    def explore_duplicate_rows(self, df):
        duplicate_rows = df[df.duplicated(keep=False)]
        duplicate_count = len(duplicate_rows)
        duplicate_percentage = (duplicate_count / len(df)) * 100

        if not duplicate_rows.empty:
            st.header("Duplicate Rows:")
            st.write(f"Number of Duplicate Rows: {duplicate_count}")
            st.write(f"Percentage of Duplicate Rows: {duplicate_percentage:.2f}%")
            st.write(duplicate_rows)

            delete_duplicates = st.checkbox("Delete Duplicate Rows")
            if delete_duplicates:
                df.drop_duplicates(keep="first", inplace=True)
                st.success("Duplicate rows have been deleted.")
        else:
            st.info("No duplicate rows found in the dataset.")

    def calculate_summary_statistics(self, df):
        summary_stats = df.describe()
        selected_stats = summary_stats.loc[['mean', '50%', 'count', 'min', 'max']]

        mode_values = df.mode().iloc[0]
        selected_stats.loc['Mode'] = mode_values

        selected_stats.rename(index={'mean': 'Mean', '50%': 'Median', 'count': 'Count', 'min': 'Min', 'max': 'Max'},
                              inplace=True)

        st.header("Summary Statistics:")
        st.write(selected_stats)

    def create_bar_graph(self, df_cleaned):
        st.title("Graphs")
        graph = st.selectbox("Type of graph", ["Select", "Bar Graph", "Histogram"])

        if graph == "Bar Graph" and df_cleaned is not None:
            st.write("Select columns for the bar graph:")
            x_columns = st.multiselect("Select X-axis Column(s)", df_cleaned.columns)
            y_columns = st.multiselect("Select Y-axis Column(s)", df_cleaned.columns)

            if x_columns and y_columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                for y_column in y_columns:
                    for x_column in x_columns:
                        sns.barplot(x=x_column, y=y_column, data=df_cleaned)
                plt.xlabel(", ".join(x_columns))
                plt.ylabel(", ".join(y_columns))
                plt.title("Bar Graph")
                st.pyplot(fig)

class MLModelHandler:
    def perform_linear_regression(self, df_cleaned, target_variable, feature_variables):
        X = df_cleaned[feature_variables]
        y = df_cleaned[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R2): {r2:.2f}")

        # Scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, c='b', label='Predicted', alpha=0.5)
        plt.scatter(y_test, y_test, c='r', label='Actual', alpha=0.5)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.legend(loc='best')
        plt.grid()
        st.pyplot(plt)

        # Coefficients plot
        if len(feature_variables) > 1:
            coefficients_df = pd.DataFrame({'Coefficient': model.coef_, 'Feature': feature_variables})
            coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)

            plt.figure(figsize=(10, 5))
            sns.barplot(x='Coefficient', y='Feature', data=coefficients_df, palette='viridis')
            plt.title("Coefficients of Selected Features")
            st.pyplot(plt)

        # Recommendations based on results
        st.subheader("Insights and Recommendations:")
        if r2 > 0.7:
            st.write("The R-squared value indicates a strong linear relationship between features and the target variable.")
            st.write("The model is performing well.")
        else:
            st.write("The R-squared value suggests that the model may benefit from feature engineering or other algorithms.")
            st.write("Consider exploring additional features or different regression models.")

        return model, mse, r2
        

    def perform_random_forest_classification(self, df_cleaned, target_variable):
        X = df_cleaned.drop(columns=[target_variable])
        y = df_cleaned[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix:")
        confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        st.write(confusion_matrix)

        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=1, linecolor='black')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

        # Recommendations based on results
        st.subheader("Insights and Recommendations:")
        if accuracy > 0.85:
            st.write("The model achieved a high accuracy, indicating good classification performance.")
            st.write("The features selected are informative for predicting the target variable.")
        else:
            st.write("Consider exploring feature engineering, hyperparameter tuning, or alternative classification algorithms.")

        return classifier, accuracy, precision, recall, f1
        

def main():
    st.set_page_config()
    st.title('Exploratory Data Analysis')
    st.write('Select a data source for analysis by uploading a CSV file or connecting to a Snowflake database.'
             ' Conduct EDA to understand the dataset\'s characteristics and relationships.'
             ' Create basic data visualizations for insights.'
             ' Choose from different machine learning models for predictive analysis')

    st.sidebar.header("Sidebar")
    data_source = st.sidebar.selectbox("Select Data Source", ["Select", "Upload CSV", "Connect to Snowflake"])

    data_handler = DataHandler()
    ml_model_handler = MLModelHandler()

    if data_source == "Upload CSV":
        df = data_handler.upload_csv()
        if df is not None:
            data_handler.display_uploaded_data(df)
            data_handler.explore_duplicate_rows(df)
            data_handler.calculate_summary_statistics(df)

            clean_option = st.selectbox("Select Data Cleaning Option", ["None", "Drop Missing Values", "Impute with Mean", "Impute with Median", "Impute with Mode"])
            df_cleaned = data_handler.clean_data(df, clean_option)
            data_handler.display_cleaned_data(df_cleaned)

            if df_cleaned is not None:
                data_handler.create_bar_graph(df_cleaned)

    elif data_source == "Connect to Snowflake":
        st.sidebar.error("Coming up with Snowflake connection functionality")
    st.sidebar.header("ML Model Selection")
    analysis_type = st.sidebar.selectbox("Analysis Type", ["Select", "Classification", "Regression"])

    model_suggestions = {
        "Classification": "Consider models like Random Forest or Naive Bayes.",
        "Regression": "You can use Linear Regression."
    }

    if analysis_type != "Select":
        tooltip_text = model_suggestions.get(analysis_type, "No suggestions available.")
        st.sidebar.text_area("Model Suggestions", tooltip_text, height=100)

        if analysis_type == "Regression":
            ml_model = st.sidebar.selectbox("Select ML Model", ["Select", "Linear Regression"])

            if ml_model == "Linear Regression":
                st.header("You selected Linear Regression")
                target_variable = st.selectbox("Select Target Variable", df_cleaned.columns)
                feature_variables = st.multiselect("Select Feature Variable(s)", df_cleaned.columns.difference([target_variable]))

                if st.button("Perform Linear Regression"):
                    model, mse, r2 = ml_model_handler.perform_linear_regression(df_cleaned, target_variable, feature_variables)
                    

        elif analysis_type == "Classification":
            ml_model = st.sidebar.selectbox("Select ML Model", ["Select", "Random Forest"])

            if ml_model == "Random Forest":
                st.header("You selected Random Forest for Classification")
                categorical_columns = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
                target_variable = st.selectbox("Select Target Variable", categorical_columns)

                if st.button("Perform Classification with Random Forest"):
                    classifier, accuracy, precision, recall, f1 = ml_model_handler.perform_random_forest_classification(df_cleaned, target_variable)
                    

if __name__ == "__main__":
    main()
