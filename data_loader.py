"""
data_loader.py
--------------
This file handles loading and preprocessing the TripAdvisor dataset.

The TripAdvisor dataset contains:
  - User ratings for hotels
  - 7 criteria ratings: Value, Rooms, Location, Cleanliness, Check-in/Front Desk, Service, Business Service
  - 1 overall rating
  - Ratings are on a scale of 1 to 5
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def load_tripadvisor_data(filepath):
    """
    Load the TripAdvisor dataset from a CSV file.

    The expected CSV file should have the following columns:
        user_id, item_id, value, rooms, location, cleanliness,
        checkin, service, business_service, overall

    Parameters:
        filepath (str): Path to the CSV dataset file.

    Returns:
        DataFrame: The loaded dataset as a pandas DataFrame.
    """
    print(f"Loading dataset from: {filepath}")

    # Read the CSV file into a DataFrame
    data = pd.read_csv(filepath)

    # Show basic info about the loaded dataset
    print(f"Dataset loaded successfully.")
    print(f"  Total number of rows (ratings): {len(data)}")
    print(f"  Columns found: {list(data.columns)}")

    return data


def encode_ids(data):
    """
    Re-encode user IDs and item IDs to be consecutive integers starting from 0.

    Deep learning embedding layers require IDs to be in the range [0, num_users)
    and [0, num_items). If the original IDs are sparse (e.g., user IDs like 1001, 5023, ...),
    we map them to 0, 1, 2, ... to keep the embedding table compact.

    Parameters:
        data (DataFrame): The raw dataset.

    Returns:
        data (DataFrame): Dataset with re-encoded IDs.
        num_users (int): Total number of unique users.
        num_items (int): Total number of unique items (hotels).
    """
    # Get all unique user IDs and item IDs
    unique_users = data['user_id'].unique()
    unique_items = data['item_id'].unique()

    # Create a mapping: original ID -> sequential index starting from 0
    user_to_index = {user: index for index, user in enumerate(unique_users)}
    item_to_index = {item: index for index, item in enumerate(unique_items)}

    # Apply the mapping to the DataFrame columns
    data = data.copy()
    data['user_id'] = data['user_id'].map(user_to_index)
    data['item_id'] = data['item_id'].map(item_to_index)

    num_users = len(unique_users)
    num_items = len(unique_items)

    print(f"  Number of unique users: {num_users}")
    print(f"  Number of unique items (hotels): {num_items}")

    return data, num_users, num_items


def get_criteria_columns():
    """
    Return the list of the 7 criteria rating column names used in the paper.

    These are the 7 sub-aspects that users rate in TripAdvisor:
        Value, Rooms, Location, Cleanliness, Check-in/Front Desk, Service, Business Service

    Returns:
        list: List of column name strings.
    """
    return ['value', 'rooms', 'location', 'cleanliness', 'checkin', 'service', 'business_service']


def split_data_kfold(data, n_splits=5, random_state=42):
    """
    Split the dataset into training and test sets using K-Fold Cross Validation.

    The paper uses 5-fold cross-validation:
        - 80% training (which includes 1% validation)
        - 20% test

    Parameters:
        data (DataFrame): The full dataset.
        n_splits (int): Number of folds (default: 5 as per the paper).
        random_state (int): Seed for reproducibility.

    Returns:
        list of tuples: Each tuple is (train_df, test_df) for one fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Store all folds as a list of (train, test) tuples
    folds = []

    # data.index gives us all row indices; KFold splits them
    for fold_number, (train_indices, test_indices) in enumerate(kf.split(data)):
        train_data = data.iloc[train_indices].reset_index(drop=True)
        test_data = data.iloc[test_indices].reset_index(drop=True)
        folds.append((train_data, test_data))
        print(f"  Fold {fold_number + 1}: {len(train_data)} training samples, {len(test_data)} test samples")

    return folds


def extract_arrays(data):
    """
    Extract numpy arrays from a DataFrame for model input/output.

    The Criteria Ratings DNN takes:
        - user_ids: array of user indices
        - item_ids: array of item indices

    And outputs:
        - criteria_ratings: 2D array of shape (N, 7) with the 7 criteria ratings

    The Overall Rating DNN takes:
        - criteria_ratings (normalized)
    And outputs:
        - overall_ratings: 1D array of overall ratings

    Parameters:
        data (DataFrame): A split (train or test) DataFrame.

    Returns:
        user_ids (np.array): Shape (N,)
        item_ids (np.array): Shape (N,)
        criteria_ratings (np.array): Shape (N, 7)
        overall_ratings (np.array): Shape (N,)
    """
    criteria_cols = get_criteria_columns()

    user_ids = data['user_id'].values.astype(np.int32)
    item_ids = data['item_id'].values.astype(np.int32)
    criteria_ratings = data[criteria_cols].values.astype(np.float32)
    overall_ratings = data['overall'].values.astype(np.float32)

    return user_ids, item_ids, criteria_ratings, overall_ratings


def create_sample_dataset(output_path, num_users=100, num_items=50, num_ratings=500):
    """
    Create a small synthetic sample dataset for testing purposes.

    This generates random user/item IDs and ratings in the range [1, 5].
    Use this if you do not have the real TripAdvisor dataset.

    Parameters:
        output_path (str): Path where the CSV will be saved.
        num_users (int): Number of simulated users.
        num_items (int): Number of simulated items (hotels).
        num_ratings (int): Number of total ratings to generate.
    """
    np.random.seed(42)
    print(f"Creating synthetic sample dataset with {num_ratings} ratings...")

    # Randomly pick user and item pairs
    user_ids = np.random.randint(0, num_users, size=num_ratings)
    item_ids = np.random.randint(0, num_items, size=num_ratings)

    # Generate random ratings between 1 and 5 for each criterion and overall
    value       = np.random.randint(1, 6, size=num_ratings).astype(float)
    rooms       = np.random.randint(1, 6, size=num_ratings).astype(float)
    location    = np.random.randint(1, 6, size=num_ratings).astype(float)
    cleanliness = np.random.randint(1, 6, size=num_ratings).astype(float)
    checkin     = np.random.randint(1, 6, size=num_ratings).astype(float)
    service     = np.random.randint(1, 6, size=num_ratings).astype(float)
    business    = np.random.randint(1, 6, size=num_ratings).astype(float)

    # The overall rating is a rough average of criteria to simulate real data
    overall = np.round(
        (value + rooms + location + cleanliness + checkin + service + business) / 7.0
    ).clip(1, 5)

    # Build a DataFrame and save to CSV
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'value': value,
        'rooms': rooms,
        'location': location,
        'cleanliness': cleanliness,
        'checkin': checkin,
        'service': service,
        'business_service': business,
        'overall': overall
    })

    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to: {output_path}")
    return df
