import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
@st.cache_data
def load_data():
    file_path = "Modified_Walmart_Sales_Data_for_BI.csv"
    return pd.read_csv(file_path)

# Initialize data
data = load_data()

# Streamlit app setup
st.title("Business Intelligence Dashboard")
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Select a feature to explore:",
    ["Dataset Overview", "EDA", "Sales Trends", "Prediction", "Advanced Insights"]
)

# Sidebar filters
st.sidebar.subheader("Filter Data")
branch_filter = st.sidebar.multiselect("Select Branch(es):", data['Branch'].unique())
product_filter = st.sidebar.multiselect("Select Product Line(s):", data['Product line'].unique())
date_filter = st.sidebar.date_input("Select Date Range:", [])

# Apply filters
filtered_data = data.copy()
if branch_filter:
    filtered_data = filtered_data[filtered_data['Branch'].isin(branch_filter)]
if product_filter:
    filtered_data = filtered_data[filtered_data['Product line'].isin(product_filter)]
if date_filter:
    if isinstance(date_filter, list) and len(date_filter) == 2:
        filtered_data = filtered_data[(filtered_data['Date'] >= pd.to_datetime(date_filter[0])) & 
                                      (filtered_data['Date'] <= pd.to_datetime(date_filter[1]))]

# Dataset Overview
if options == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("Filtered Dataset:")
    st.write(filtered_data.head())
    st.write("Shape of the dataset:", filtered_data.shape)

    # Basic statistics
    st.subheader("Summary Statistics")
    st.write(filtered_data.describe())

# Exploratory Data Analysis (EDA)
if options == "EDA":
    st.header("Exploratory Data Analysis")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_data = filtered_data.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pareto Analysis
    st.subheader("Pareto Analysis: Top Products")
    sales_by_product = filtered_data.groupby('Product line')['Total'].sum().reset_index()
    sales_by_product['Cumulative Percentage'] = sales_by_product['Total'].cumsum() / sales_by_product['Total'].sum() * 100

    fig = px.bar(sales_by_product, x='Product line', y='Total', text='Cumulative Percentage',
                 title="Pareto Analysis of Sales by Product Line")
    st.plotly_chart(fig)

# Sales Trends
if options == "Sales Trends":
    st.header("Sales Trends")

    # Convert date column
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
    sales_trend = filtered_data.groupby('Date')['Total'].sum().reset_index()

    # Sales trends plot
    fig = px.line(sales_trend, x='Date', y='Total', title='Sales Over Time', labels={"Date": "Date", "Total": "Total Sales"})
    st.plotly_chart(fig)

    # Seasonality Analysis
    st.subheader("Seasonality Analysis")
    filtered_data['Month'] = filtered_data['Date'].dt.month
    seasonality = filtered_data.groupby('Month')['Total'].mean().reset_index()

    fig = px.line(seasonality, x='Month', y='Total', title="Average Monthly Sales",
                  labels={"Month": "Month", "Total": "Average Sales"})
    st.plotly_chart(fig)

# Predictive Analytics
if options == "Prediction":
    st.header("Predict Future Sales")

    # Ensure the 'Date' column is datetime
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
    sales_trend = filtered_data.groupby('Date')['Total'].sum().reset_index()
    sales_trend['Day'] = sales_trend['Date'].dt.dayofyear

    # Prepare data for prediction
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

    # Plot future sales
    fig = px.line(x=future_days['Day'], y=future_sales, title='Predicted Sales for Next Year',
                  labels={"x": "Day of Year", "y": "Predicted Sales"})
    st.plotly_chart(fig)

# Advanced Insights
if options == "Advanced Insights":
    st.header("Advanced Insights")

    # Gross Margin Analysis
    st.subheader("Gross Margin Analysis")
    filtered_data['Gross Margin'] = (filtered_data['Total'] - filtered_data['cogs']) / filtered_data['Total'] * 100
    margin_summary = filtered_data.groupby('Product line')['Gross Margin'].mean().reset_index()

    fig = px.bar(margin_summary, x='Product line', y='Gross Margin', title="Average Gross Margin by Product Line",
                 labels={"Gross Margin": "Gross Margin (%)"})
    st.plotly_chart(fig)

    # RFM Analysis using 'Customer type'
    st.subheader("Customer Segmentation (RFM Analysis)")
    rfm = filtered_data.groupby('Customer type').agg(
        Recency=('Date', lambda x: (pd.to_datetime('today') - pd.to_datetime(x).max()).days),
        Frequency=('Invoice ID', 'count'),  # Assuming 'Invoice ID' indicates transactions
        Monetary=('Total', 'sum')
    ).reset_index()

    # Visualize the segmentation
    fig = px.scatter(rfm, x='Recency', y='Frequency', size='Monetary',
                     color='Customer type', title="Customer Segmentation (RFM Analysis)",
                     labels={"Recency": "Days Since Last Purchase",
                             "Frequency": "Purchase Frequency",
                             "Monetary": "Total Monetary Value"})
    st.plotly_chart(fig)
else:
        st.write("The dataset does not contain 'Custom type' information for segmentation.")

# Profitability Prediction
if options == "Profitability Prediction":
    st.header("Profitability Prediction")

    # Feature engineering
    data['Branch_Encoded'] = LabelEncoder().fit_transform(data['Branch'])
    X = data[['Total', 'cogs', 'Branch_Encoded']]
    y = data['Profit Margin %']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
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

# Customer Behavior Prediction
if options == "Customer Behavior Prediction":
    st.header("Customer Behavior Prediction")

    # Feature engineering
    data['Customer_Type_Encoded'] = LabelEncoder().fit_transform(data['Customer_Type'])
    X = data[['Total', 'Unit price', 'Quantity']]
    y = data['Customer_Type_Encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy for Customer Behavior Prediction: {accuracy:.2f}")

    # Confusion matrix visualization
    st.subheader("Confusion Matrix")
    conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    st.write(conf_matrix)