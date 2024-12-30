import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Streamlit App Title
st.title("Slack Channel Analytics Dashboard")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV file containing Slack channel analytics", type=["csv"])

if uploaded_file:
    # Load the data
    data = pd.read_csv(uploaded_file)

    # Convert the 'Created' column to datetime
    data['Created'] = pd.to_datetime(data['Created'], errors='coerce')

    # Convert pandas Timestamps to Python datetime for Streamlit compatibility
    min_date = data['Created'].min().to_pydatetime() if pd.notnull(data['Created'].min()) else None
    max_date = data['Created'].max().to_pydatetime() if pd.notnull(data['Created'].max()) else None

    # Display data preview
    st.subheader("Data Preview")
    st.write(data.head())

    # Interactive Slider: Filter by creation date
    st.subheader("Filter Channels by Creation Date")
    if min_date and max_date:
        date_filter = st.slider(
            "Select a date range to filter channels:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )

        # Filter data based on the slider input
        filtered_data = data[(data['Created'] >= date_filter[0]) & (data['Created'] <= date_filter[1])]

        # Scatterplot: Date Created vs. Messages Posted
        st.subheader("Scatterplot: Date Created vs. Number of Messages")
        if not filtered_data.empty:
            plt.figure(figsize=(10, 5))
            plt.scatter(filtered_data['Created'], filtered_data['Messages posted'], alpha=0.7, color='green', edgecolor='black')
            plt.xlabel("Date Created")
            plt.ylabel("Number of Messages Posted")
            plt.title("Channels: Date Created vs. Messages Posted")
            st.pyplot(plt)
        else:
            st.write("No channels match the selected date range.")
    else:
        st.write("Invalid date data in the dataset.")
