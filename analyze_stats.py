import sqlite3
import sys
from collections import Counter
import pandas as pd # Using pandas for easier aggregation and display

# --- Configuration ---
db_file = "processed_ids.sqlite"

# --- Main Analysis Logic ---
if __name__ == "__main__":
    conn = None
    try:
        print(f"Connecting to database: {db_file}")
        # Use check_same_thread=False if you might run analysis concurrently later
        # Use uri=True for more connection options if needed
        conn = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True) # Connect in read-only mode
        cur = conn.cursor()
        print("Connection successful.")

        # --- Fetch all relevant data --- 
        print("Fetching data from processed_assets table...")
        try:
            query = "SELECT asset_id, matched_keyword, similarity_score, target_album_id, target_album_name, processed_at FROM processed_assets"
            df = pd.read_sql_query(query, conn)
            print(f"Fetched {len(df)} log entries.")
        except pd.io.sql.DatabaseError as e:
             if "no such table: processed_assets" in str(e):
                 print(f"Error: The table 'processed_assets' does not exist in {db_file}. Did the main script run successfully?")
             else:
                 print(f"Error reading data from database: {e}")
             sys.exit(1)

        if df.empty:
            print("No data found in the processed_assets table. Exiting.")
            sys.exit(0)

        # Ensure correct data types (optional but good practice)
        # Use errors='coerce' to handle potential invalid data gracefully
        original_rows = len(df)
        
        # Store original potentially problematic columns before coercion
        original_scores = df['similarity_score'].copy()
        original_dates = df['processed_at'].copy()
        
        df['processed_at'] = pd.to_datetime(df['processed_at'], errors='coerce')
        df['similarity_score'] = pd.to_numeric(df['similarity_score'], errors='coerce')

        # --- Report rows with conversion errors ---
        invalid_date_mask = df['processed_at'].isna()
        invalid_score_mask = df['similarity_score'].isna()
        
        if invalid_date_mask.any():
             invalid_date_rows_indices = df.index[invalid_date_mask]
             print(f"\nWarning: Found {len(invalid_date_rows_indices)} rows with invalid 'processed_at' dates. Examples (Original Values):")
             # Show original dates from before coercion
             report_df_dates = pd.DataFrame({
                 'asset_id': df.loc[invalid_date_mask, 'asset_id'],
                 'matched_keyword': df.loc[invalid_date_mask, 'matched_keyword'],
                 'original_processed_at': original_dates[invalid_date_mask]
             })
             print(report_df_dates.head())

        if invalid_score_mask.any():
             invalid_score_rows_indices = df.index[invalid_score_mask]
             print(f"\nWarning: Found {len(invalid_score_rows_indices)} rows with invalid 'similarity_score' values. Examples (Original Values):")
             # Show original scores from before coercion
             report_df_scores = pd.DataFrame({
                 'asset_id': df.loc[invalid_score_mask, 'asset_id'],
                 'matched_keyword': df.loc[invalid_score_mask, 'matched_keyword'],
                 'original_similarity_score': original_scores[invalid_score_mask]
             })
             print(report_df_scores.head())

        # --- Filter out rows with conversion errors before analysis ---
        # Use the masks we already calculated
        valid_rows_mask = ~invalid_date_mask & ~invalid_score_mask
        df = df[valid_rows_mask].copy() # Create a copy to avoid SettingWithCopyWarning
        
        rows_after_cleaning = len(df)
        if rows_after_cleaning < original_rows:
            print(f"\nNote: Removed {original_rows - rows_after_cleaning} rows due to data conversion errors before analysis.")
            if df.empty:
                print("No valid data remaining after cleaning. Exiting.")
                sys.exit(0)

        # --- Calculate Statistics ---
        print("\n--- Analysis Results ---")

        # 1. Total Unique Assets Processed
        total_unique_assets = df['asset_id'].nunique()
        print(f"Total unique assets processed: {total_unique_assets}")

        # 2. Counts per Matched Keyword/Group
        # This counts the number of log entries for each keyword/group.
        # If an asset is added to multiple albums by the same keyword/group, it's counted multiple times here.
        keyword_counts = df['matched_keyword'].value_counts().sort_index()
        print("\nAssets processed per Keyword/Group (Log Entry Count):")
        print(keyword_counts.to_string())
        # Alternative: Count *unique* assets per keyword/group
        # unique_assets_per_keyword = df.groupby('matched_keyword')['asset_id'].nunique().sort_index()
        # print("\nUnique assets processed per Keyword/Group:")
        # print(unique_assets_per_keyword.to_string())

        # 3. Total Unique Albums Involved
        total_unique_albums = df['target_album_name'].nunique()
        print(f"\nTotal unique target albums involved: {total_unique_albums}")

        # 4. Counts per Target Album
        # Counts how many times assets were added to each specific album
        album_counts = df['target_album_name'].value_counts().sort_index()
        print("\nAssets added per Target Album (Log Entry Count):")
        print(album_counts.to_string())

        # 5. Basic Similarity Statistics per Keyword/Group
        # Note: Similarity score for groups is the summed score.
        similarity_stats = df.groupby('matched_keyword')['similarity_score'].agg(['count', 'mean', 'min', 'max', 'median', 'std'])
        print("\nSimilarity Score Statistics per Keyword/Group:")
        # Format output for better readability
        pd.options.display.float_format = '{:.3f}'.format
        print(similarity_stats)

        # Add more stats here if desired
        # Example: Timeline analysis (e.g., assets processed per day)
        # df['processed_date'] = df['processed_at'].dt.date
        # daily_counts = df.groupby('processed_date')['asset_id'].nunique()
        # print("\nUnique Assets Processed Per Day:")
        # print(daily_counts.to_string())


    except sqlite3.OperationalError as e:
        if "unable to open database file" in str(e):
             print(f"Error: Could not open database file '{db_file}'. Does it exist?")
        elif "database is locked" in str(e):
             print(f"Error: Database file '{db_file}' is locked. Is the main script running?")
        else:
             print(f"SQLite Operational Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
         print(f"Error: Database file '{db_file}' not found.")
         sys.exit(1)
    except ImportError:
        print("Error: Pandas library not found. Please install it: pip install pandas")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.") 