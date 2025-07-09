import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('hr_dataset_switzerland.csv')
    return data

# Main app
def main():
    st.title('HR Dataset Switzerland')

    # Load data
    data = load_data()

    # Display raw data
    st.subheader('Raw Data')
    st.write(data)

    # Basic data info
    st.subheader('Data Info')
    st.write(data.describe())

    # Simple query example
    st.subheader('Query Data')
    column = st.selectbox('Select a column to query', data.columns)
    unique_values = data[column].unique()
    selected_value = st.selectbox('Select a value', unique_values)

    # Filter data
    filtered_data = data[data[column] == selected_value]
    st.write(filtered_data)

    # Basic calculations
    st.subheader('Basic Calculations')
    if pd.api.types.is_numeric_dtype(filtered_data[column]):
        st.write(f"Mean of {column}: {filtered_data[column].mean()}")
    st.write(f"Count of {column}: {filtered_data[column].count()}")

    # Simple chart
    st.subheader('Chart')
    chart_type = st.selectbox('Select chart type', ['Line', 'Bar', 'Histogram'])

    if chart_type == 'Line':
        st.line_chart(filtered_data[[column]])
    elif chart_type == 'Bar':
        st.bar_chart(filtered_data[[column]])
    elif chart_type == 'Histogram':
        if pd.api.types.is_numeric_dtype(filtered_data[column]):
            st.write(plt.hist(filtered_data[column]))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        else:
            st.write("Histogram can only be plotted for numeric data.")

if __name__ == '__main__':
    main()
