import pandas as pd
from pathlib import Path
import argparse
import geopandas as gpd
from shapely.geometry import Point

def engineer_features(df):
    
    df = df.copy()

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    df = df.rename(columns={'trans_date_trans_time': 'trans_datetime'})

    df['merchant'] = df['merchant'].str.removeprefix('fraud_')

    df['trans_datetime'] = pd.to_datetime(df['trans_datetime'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    df['hour'] = df['trans_datetime'].dt.hour
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek # Monday=0, Sunday=6
    df['month'] = df['trans_datetime'].dt.month
    
    df['age'] = (df['trans_datetime'] - df['dob']).dt.days // 365

    cardholder_geom = gpd.GeoSeries.from_xy(df['long'], df['lat'])
    merchant_geom = gpd.GeoSeries.from_xy(df['merch_long'], df['merch_lat'])
    distance_deg = cardholder_geom.distance(merchant_geom)
    df['distance_km'] = distance_deg * 111

    df['bin'] = df['cc_num'].astype(str).str[:6]


    columns_to_drop = [
        'trans_datetime', 'dob', 'lat', 'long', 'merch_lat', 'merch_long',
        'cc_num', 'first', 'last', 'street', 'city', 'zip', 'job', 
        'merchant', 'trans_num'
    ]
    df = df.drop(columns=columns_to_drop)
    
    return df


def main(args):

    print("-> Starting Data Preprocessing ")
    df_train_raw = pd.read_csv(args.train_input)
    df_test_raw = pd.read_csv(args.test_input)
    
    print("Engineering features for training data...")
    df_train_processed = engineer_features(df_train_raw)
    
    print("Engineering features for test data...")
    df_test_processed = engineer_features(df_test_raw)

    cols_to_cap = 'amt'
    
    #for col in cols_to_cap:
    cap_value = df_train_processed[cols_to_cap].quantile(0.99)
    print(f"Capping '{cols_to_cap}' at 99th percentile: {cap_value:.2f}")

    df_train_processed[cols_to_cap] = df_train_processed[cols_to_cap].clip(upper=cap_value)
    df_test_processed[cols_to_cap] = df_test_processed[cols_to_cap].clip(upper=cap_value)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_processed.csv"
    test_path = output_dir / "test_processed.csv"
    df_train_processed.to_csv(train_path, index=False)
    df_test_processed.to_csv(test_path, index=False)

    print(f"--- Data Preprocessing Complete. Files saved to {train_path} and {test_path} ---")
    print("\nStatistical summary of capped training data:")
    print(df_train_processed[cols_to_cap].describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the raw fraud detection data.")
    
    parser.add_argument("--train-input", type=str, required=True, help="Path to the raw training data CSV file.")
    parser.add_argument("--test-input", type=str, required=True, help="Path to the raw test data CSV file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the processed train and test CSV files.")

    args = parser.parse_args()
    main(args)