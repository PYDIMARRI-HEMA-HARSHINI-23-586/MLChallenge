import pandas as pd
import numpy as np
import os

# Define the path to the dataset
train_file = 'C:\\Users\\HEMA HARSHINI\\Desktop\\ml\\MLChallenge\\68e8d1d70b66d_student_resource\\student_resource\\dataset\\train.csv'

# Load the training data
try:
    train_df = pd.read_csv(train_file)

    # Display the first 5 rows
    print("First 5 rows of the training data:")
    print(train_df.head())
    print("\n" + "="*50 + "\n")

    # Display concise summary of the dataframe
    print("Dataframe Info:")
    # Use a string buffer to capture the info output
    import io
    buf = io.StringIO()
    train_df.info(buf=buf)
    info_str = buf.getvalue()
    print(info_str)
    print("\n" + "="*50 + "\n")

    # Display statistics for the 'price' column
    print("Descriptive statistics for the 'price' column:")
    print(train_df['price'].describe())
    print("\n" + "="*50 + "\n")

    # Check for missing values
    print("Missing values in each column:")
    print(train_df.isnull().sum())
    print("\n" + "="*50 + "\n")

    # Look at the distribution of the price
    price_stats = train_df['price'].describe()
    print("Price Distribution Analysis:")
    print(f"The prices range from ${price_stats['min']:.2f} to ${price_stats['max']:.2f}.")
    print(f"The average price is ${price_stats['mean']:.2f}.")
    print(f"The median price is ${price_stats['50%']:.2f}.")
    if price_stats['mean'] > price_stats['50%']:
        print("The distribution of prices appears to be right-skewed, meaning there are some products with very high prices.")
    else:
        print("The price distribution is more symmetric.")


except FileNotFoundError:
    print(f"Error: {train_file} not found. Please check the file path.")
