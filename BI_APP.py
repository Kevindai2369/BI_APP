import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_data
def load_data():
    file_path = "Modified_Walmart_Sales_Data_for_BI.csv"
    data = pd.read_csv(file_path)
    return preprocess_data(data)

# Preprocessing function
def preprocess_data(data):
    # Handle missing values for numeric columns
    num_cols = ['Unit_Price', 'Quantity', 'Tax_5', 'Total', 'cogs', 'Gross_Income', 'Rating']
    for col in num_cols:
        if col in data.columns:
            data[col].fillna(data[col].mean(), inplace=True)

    # Handle missing values for categorical columns
    cat_cols = ['Invoice_ID', 'Branch', 'City', 'Customer_Type', 'Gender', 'Product_Line', 'Payment']
    for col in cat_cols:
        if col in data.columns:
            data[col].fillna("Unknown", inplace=True)

    # Convert 'Date' to datetime format
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Extract Year, Month, and Day for easier analysis
    if 'Date' in data.columns:
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day

    return data

# Load and initialize the data
data = load_data()

# Verify dataset columns
st.write("Columns in the dataset:", data.columns)

# Streamlit app setup
st.title("Business Intelligence Dashboard")
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Select a feature to explore:",
    ["Dataset Overview", "EDA", "Sales Trends", "Prediction", "Advanced Insights", "Profitability Prediction", "Customer Behavior Prediction"]
)

# Sidebar filters
st.sidebar.subheader("Filter Data")
branch_filter = st.sidebar.multiselect("Select Branch(es):", data['Branch'].unique() if 'Branch' in data.columns else [])
product_filter = st.sidebar.multiselect("Select Product Line(s):", data['Product_Line'].unique() if 'Product_Line' in data.columns else [])
date_filter = st.sidebar.date_input("Select Date Range:", [])

# Apply filters
filtered_data = data.copy()
if branch_filter:
    filtered_data = filtered_data[filtered_data['Branch'].isin(branch_filter)]
if product_filter:
    filtered_data = filtered_data[filtered_data['Product_Line'].isin(product_filter)]
if date_filter:
    if isinstance(date_filter, list) and len(date_filter) == 2 and 'Date' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['Date'] >= pd.to_datetime(date_filter[0])) &
                                      (filtered_data['Date'] <= pd.to_datetime(date_filter[1]))]

# Dataset Overview
if options == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("Filtered Dataset:")
    st.write(filtered_data.head())
    st.write("Shape of the dataset:", filtered_data.shape)
    st.subheader("Summary Statistics")
    st.write(filtered_data.describe())

# EDA
if options == "EDA":
    st.header("Exploratory Data Analysis")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_data = filtered_data.select_dtypes(include=['number'])
    if not numeric_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric data available for correlation analysis.")

    # Pareto Analysis
    st.subheader("Pareto Analysis: Top Products")
    if 'Product_Line' in filtered_data.columns and 'Total' in filtered_data.columns:
        sales_by_product = filtered_data.groupby('Product_Line')['Total'].sum().reset_index()
        sales_by_product['Cumulative Percentage'] = sales_by_product['Total'].cumsum() / sales_by_product['Total'].sum() * 100
        fig = px.bar(sales_by_product, x='Product_Line', y='Total', text='Cumulative Percentage',
                     title="Pareto Analysis of Sales by Product Line")
        st.plotly_chart(fig)
    else:
        st.write("Required columns for Pareto Analysis are missing.")

# Sales Trends
if options == "Sales Trends":
    st.header("Sales Trends")

    if 'Date' in filtered_data.columns and 'Total' in filtered_data.columns:
        # Sales trends plot
        sales_trend = filtered_data.groupby('Date')['Total'].sum().reset_index()
        fig = px.line(sales_trend, x='Date', y='Total', title='Sales Over Time', labels={"Date": "Date", "Total": "Total Sales"})
        st.plotly_chart(fig)

        # Seasonality Analysis
        st.subheader("Seasonality Analysis")
        filtered_data['Month'] = filtered_data['Date'].dt.month
        seasonality = filtered_data.groupby('Month')['Total'].mean().reset_index()
        fig = px.line(seasonality, x='Month', y='Total', title="Average Monthly Sales",
                      labels={"Month": "Month", "Total": "Average Sales"})
        st.plotly_chart(fig)
    else:
        st.write("Date or Total column is missing for sales trend analysis.")

# Prediction
if options == "Prediction":
    st.header("Predict Future Sales")

    if 'Date' in filtered_data.columns and 'Total' in filtered_data.columns:
        sales_trend = filtered_data.groupby('Date')['Total'].sum().reset_index()
        sales_trend['Day'] = sales_trend['Date'].dt.dayofyear
        X = sales_trend[['Day']]
        y = sales_trend['Total']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        model_option = st.selectbox("Select Prediction Model:", ["Linear Regression", "Random Forest", "Support Vector Machine"])
        if model_option == "Linear Regression":
            model = LinearRegression()
        elif model_option == "Random Forest":
            model = RandomForestRegressor()
        elif model_option == "Support Vector Machine":
            model = SVR()

        # Train and evaluate model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error using {model_option}: {mse:.2f}")

        # Future predictions
        future_days = pd.DataFrame({'Day': range(1, 366)})
        future_sales = model.predict(future_days)
        fig = px.line(x=future_days['Day'], y=future_sales, title='Predicted Sales for Next Year',
                      labels={"x": "Day of Year", "y": "Predicted Sales"})
        st.plotly_chart(fig)
    else:
        st.write("Date or Total column is missing for prediction.")

# Advanced Insights
if options == "Advanced Insights":
    st.header("Advanced Insights")

    # Check for required columns
    required_columns = ['Product_Line', 'Gross_Margin_Percentage']
    missing_columns = [col for col in required_columns if col not in filtered_data.columns]

    if not missing_columns:
        st.subheader("Gross Margin Analysis")

        # Group by Product Line and calculate average Gross Margin
        margin_summary = filtered_data.groupby('Product_Line')['Gross_Margin_Percentage'].mean().reset_index()

        # Plot the Gross Margin Analysis
        fig = px.bar(
            margin_summary,
            x='Product_Line',
            y='Gross_Margin_Percentage',
            title="Average Gross Margin by Product Line",
            labels={"Gross_Margin_Percentage": "Gross Margin (%)", "Product_Line": "Product Line"}
        )
        st.plotly_chart(fig)
    else:
        # Display missing columns
        st.error(f"Cannot perform Gross Margin Analysis. Missing column(s): {', '.join(missing_columns)}")


# Profitability Prediction
if options == "Profitability Prediction":
    st.header("Profitability Prediction")

    if 'Total' in data.columns and 'cogs' in data.columns:
        data['Profit Margin %'] = (data['Total'] - data['cogs']) / data['cogs'] * 100
        data['Branch_Encoded'] = LabelEncoder().fit_transform(data['Branch'])

        X = data[['Total', 'cogs', 'Branch_Encoded']]
        y = data['Profit Margin %']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training and prediction
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error for Profitability Prediction: {mse:.2f}")

        # Feature importance visualization
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig = px.bar(importance, x='Feature', y='Importance', title='Feature Importance')
        st.plotly_chart(fig)
    else:
        st.write("Required columns for Profitability Prediction are missing.")

# Customer Behavior Prediction
if options == "Customer Behavior Prediction":
    st.header("Customer Behavior Prediction")

    # Ensure required columns exist
    required_columns = ['Customer_Type', 'Total', 'Unit_Price', 'Quantity']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if not missing_columns:
        # Encode 'Customer_Type' column
        data['Customer_Type_Encoded'] = LabelEncoder().fit_transform(data['Customer_Type'])

        # Prepare features and target variable
        X = data[['Total', 'Unit_Price', 'Quantity']]
        y = data['Customer_Type_Encoded']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training and evaluation
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Display accuracy
        st.write(f"Accuracy for Customer Behavior Prediction: {accuracy:.2f}")

        # Confusion matrix visualization
        st.subheader("Confusion Matrix")
        conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        st.write(conf_matrix)
    else:
        # Display missing columns
        st.error(f"The dataset is missing required columns: {', '.join(missing_columns)}")
        st.write("Please ensure the following columns are present in the dataset:")
        st.write(", ".join(required_columns))