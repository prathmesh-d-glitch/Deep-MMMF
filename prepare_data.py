"""
prepare_data.py
---------------
A standalone helper script to download and preprocess the TripAdvisor dataset.

The TripAdvisor dataset used in the paper is not publicly available as a
standard download. This script provides two options:

OPTION 1: Convert a locally available TripAdvisor-style JSON/XML file
          to the CSV format expected by main.py.

OPTION 2: Use a similar publicly available dataset: the "Yahoo! Webscope
          Hotel Recommendation Dataset" or similar.

OPTION 3 (DEFAULT): Generate a realistic synthetic dataset that mimics the
          statistics reported in the paper (Table 1, Table 2) for testing
          without the real data.

The expected CSV format for main.py has these exact columns:
    user_id, item_id, value, rooms, location, cleanliness,
    checkin, service, business_service, overall
"""

import numpy as np
import pandas as pd


def generate_realistic_synthetic_data(output_path='tripadvisor_synthetic.csv',
                                       num_users=500,
                                       num_items=200,
                                       num_ratings=5000,
                                       seed=42):
    """
    Generate a synthetic dataset that mimics the TripAdvisor dataset structure
    and rough rating distribution from Table 2 of the paper.

    The paper's dataset has:
      - 72,119 users
      - 1,850 hotels
      - 81,085 ratings
      - Ratings range: 1 to 5
      - Distribution skewed towards 4 and 5 (users tend to rate positively)

    For quick testing, we use smaller numbers but keep the rating distribution
    skewed towards higher values (matching Table 2 where ratings 4 and 5 dominate).

    Parameters:
        output_path  (str): Where to save the CSV file.
        num_users    (int): Number of simulated users (paper: 72,119).
        num_items    (int): Number of simulated hotels (paper: 1,850).
        num_ratings  (int): Total number of ratings to generate (paper: 81,085).
        seed         (int): Random seed for reproducibility.

    Returns:
        DataFrame: The generated dataset.
    """
    np.random.seed(seed)
    print(f"Generating synthetic TripAdvisor-like dataset...")
    print(f"  Users: {num_users}, Hotels: {num_items}, Ratings: {num_ratings}")

    # ----- Rating distribution from paper's Table 2 -----
    # The paper shows that ratings 4 and 5 are most common (skewed distribution)
    # We model this with probabilities weighted towards higher ratings
    rating_probs = [0.05, 0.07, 0.13, 0.30, 0.45]  # Prob for ratings 1, 2, 3, 4, 5
    rating_values = [1, 2, 3, 4, 5]

    # ----- Generate user-item pairs -----
    # Each row represents one user rating one hotel
    user_ids = np.random.randint(0, num_users, size=num_ratings)
    item_ids = np.random.randint(0, num_items, size=num_ratings)

    # ----- Generate 7 criteria ratings independently -----
    # Real data has correlated criteria (e.g., cleanliness and service often correlate),
    # but for simplicity we draw each independently from the skewed distribution.
    def draw_ratings(n):
        return np.random.choice(rating_values, size=n, p=rating_probs).astype(float)

    value       = draw_ratings(num_ratings)
    rooms       = draw_ratings(num_ratings)
    location    = draw_ratings(num_ratings)
    cleanliness = draw_ratings(num_ratings)
    checkin     = draw_ratings(num_ratings)
    service     = draw_ratings(num_ratings)
    business    = draw_ratings(num_ratings)

    # ----- Overall rating is correlated with criteria average -----
    # Add a small noise term to make the relationship non-trivial
    # (so the DNN has something non-trivial to learn)
    raw_mean = (value + rooms + location + cleanliness + checkin + service + business) / 7.0
    noise    = np.random.normal(0, 0.3, size=num_ratings)
    overall  = np.round(raw_mean + noise).clip(1, 5)

    # ----- Build the DataFrame -----
    df = pd.DataFrame({
        'user_id':          user_ids,
        'item_id':          item_ids,
        'value':            value,
        'rooms':            rooms,
        'location':         location,
        'cleanliness':      cleanliness,
        'checkin':          checkin,
        'service':          service,
        'business_service': business,
        'overall':          overall
    })

    # ----- Remove duplicate (user, item) pairs - keep last -----
    # In real data, each (user, hotel) pair should have at most one review
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    df = df.reset_index(drop=True)

    print(f"  After removing duplicate (user, item) pairs: {len(df)} ratings remain")
    print(f"  Data sparsity: {1 - len(df) / (num_users * num_items):.4%}")

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Print rating distribution to compare with Table 2 of the paper
    print(f"\n  Rating distribution (compare with paper's Table 2):")
    criteria_cols = ['value', 'rooms', 'location', 'cleanliness', 'checkin', 'service', 'business_service', 'overall']
    for col in criteria_cols:
        counts = df[col].value_counts().sort_index()
        count_str = ' | '.join([f"{int(r)}: {int(counts.get(r, 0))}" for r in [1, 2, 3, 4, 5]])
        print(f"    {col:<20}: {count_str}")

    return df


def convert_json_to_csv(json_path, output_path):
    """
    Convert a TripAdvisor-style JSON file to the CSV format needed by main.py.

    This is for users who have downloaded TripAdvisor data in JSON format.
    The JSON is expected to have a list of review objects with the following fields
    (adjust field names as needed for your specific JSON structure):

    Expected JSON structure:
    [
      {
        "Author": "user123",
        "Offerings": "hotel456",
        "Ratings": {
          "Value":            4.0,
          "Rooms":            5.0,
          "Location":         3.0,
          "Cleanliness":      5.0,
          "Check in / front desk": 4.0,
          "Service":          5.0,
          "Business service": 3.0,
          "Overall":          4.0
        }
      },
      ...
    ]

    Parameters:
        json_path   (str): Path to the input JSON file.
        output_path (str): Where to save the output CSV file.

    Returns:
        DataFrame or None.
    """
    import json

    print(f"Loading JSON from: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

    print(f"Found {len(reviews)} reviews. Converting...")

    rows = []
    skipped = 0

    for review in reviews:
        try:
            # Extract user ID, item ID, and ratings
            # Adjust the key names based on your actual JSON structure
            user_id = review.get('Author', review.get('user_id', None))
            item_id = review.get('Offerings', review.get('item_id', None))

            ratings = review.get('Ratings', review.get('ratings', {}))

            # Map rating fields - handle variations in capitalization/naming
            row = {
                'user_id':          str(user_id),
                'item_id':          str(item_id),
                'value':            float(ratings.get('Value', ratings.get('value', 0))),
                'rooms':            float(ratings.get('Rooms', ratings.get('rooms', 0))),
                'location':         float(ratings.get('Location', ratings.get('location', 0))),
                'cleanliness':      float(ratings.get('Cleanliness', ratings.get('cleanliness', 0))),
                'checkin':          float(ratings.get('Check in / front desk',
                                         ratings.get('Check in / Front desk',
                                         ratings.get('checkin', 0)))),
                'service':          float(ratings.get('Service', ratings.get('service', 0))),
                'business_service': float(ratings.get('Business service',
                                         ratings.get('business_service', 0))),
                'overall':          float(ratings.get('Overall', ratings.get('overall', 0)))
            }

            # Skip rows where any rating is 0 (missing data)
            all_ratings = [row['value'], row['rooms'], row['location'], row['cleanliness'],
                           row['checkin'], row['service'], row['business_service'], row['overall']]
            if any(r == 0 for r in all_ratings):
                skipped += 1
                continue

            rows.append(row)

        except (KeyError, TypeError, ValueError) as e:
            skipped += 1
            continue

    print(f"  Successfully converted: {len(rows)} reviews")
    print(f"  Skipped (missing data): {skipped} reviews")

    if len(rows) == 0:
        print("  No valid rows found. Check the JSON structure and field names.")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Prepare TripAdvisor dataset")
    parser.add_argument('--mode', choices=['synthetic', 'convert_json'],
                        default='synthetic',
                        help='Data preparation mode')
    parser.add_argument('--json_path',   type=str, default=None,
                        help='Path to JSON input file (for convert_json mode)')
    parser.add_argument('--output_path', type=str, default='tripadvisor_synthetic.csv',
                        help='Output CSV file path')
    parser.add_argument('--num_users',   type=int, default=500)
    parser.add_argument('--num_items',   type=int, default=200)
    parser.add_argument('--num_ratings', type=int, default=5000)
    args = parser.parse_args()

    if args.mode == 'synthetic':
        generate_realistic_synthetic_data(
            output_path=args.output_path,
            num_users=args.num_users,
            num_items=args.num_items,
            num_ratings=args.num_ratings
        )
    elif args.mode == 'convert_json':
        if args.json_path is None:
            print("Error: --json_path is required for convert_json mode")
        else:
            convert_json_to_csv(args.json_path, args.output_path)
