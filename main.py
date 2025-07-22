import os
import psycopg2
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import json
import time # Import time for potential future use
import sqlite3
import argparse # Add argparse

# --- Helper Functions ---

def get_text_embeddings(texts: list[str], ml_api_url: str, clip_model: str) -> dict[str, np.ndarray] | None:
    """Fetches text embeddings from the remote ML service for a list of texts.

    Returns a dictionary mapping text -> embedding, or None on failure.
    """
    endpoint = f"{ml_api_url.rstrip('/').replace('/api','')}/predict"
    embeddings_map = {}
    print(f"Requesting embeddings for {len(texts)} unique keywords...")
    
    for text_input in texts:
        request_config = {"clip": {"textual": {"modelName": clip_model}}}
        form_data = {
            "entries": (None, json.dumps(request_config)), # (filename, content)
            "text": (None, text_input)
        }
        
        # Reduce verbosity in main script compared to debug
        # print(f"  Sending request to ML service for text: '{text_input}'") 
        try:
            # Use files parameter for multipart/form-data, even without actual files
            response = requests.post(endpoint, files=form_data) 
            response.raise_for_status() 
            
            data = response.json()
            
            # Assuming the response structure contains the embedding directly 
            # or within a predictable key. The actual response seems to be {"clip": "[num, num, ...]"}
            if isinstance(data, dict) and "clip" in data and isinstance(data["clip"], str):
                 embedding_string = data["clip"]
                 embedding_list = json.loads(embedding_string)
            else:
                print(f"Error: Unexpected response format from ML predict for '{text_input}': {data}")
                # Fail if any keyword fails for simplicity in the main script
                return None 
            
            embedding = np.array(embedding_list, dtype=np.float32)

            # Basic validation of embedding dimensions (adjust if needed)
            expected_dim = 1024 
            if embedding.shape[0] != expected_dim: 
                 print(f"Warning: ML service returned embedding with unexpected shape (Expected {expected_dim}) for '{text_input}': {embedding.shape}")
                 # Decide if this is fatal? For now, let's allow it but warn.
            
            embeddings_map[text_input] = embedding
            # Add a small delay to avoid overwhelming the ML service? 
            # time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error contacting ML service at {endpoint} for '{text_input}': {e}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON response from ML service for '{text_input}': {response.text}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred fetching text embedding for '{text_input}': {e}")
            return None
            
    if len(embeddings_map) != len(texts):
         # This condition might be less likely now since we return None earlier on error
         print("Error: Failed to get embeddings for all requested texts.")
         return None
         
    print(f"Successfully fetched embeddings for {len(embeddings_map)} keywords.")
    return embeddings_map

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates the cosine similarity between two numpy vectors."""
    if vec1.shape != vec2.shape:
        # This shouldn't happen if validation is correct, but safety first
        print(f"Warning: Shape mismatch for similarity calc: {vec1.shape} vs {vec2.shape}")
        return 0.0 
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0 # Avoid division by zero
        
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

# --- Immich API Helpers ---

def get_immich_album_id(api_url: str, api_key: str, album_name: str) -> str | None:
    """Gets the ID of an Immich album by its name."""
    endpoint = f"{api_url.rstrip('/')}/albums"
    headers = {'x-api-key': api_key}
    try:
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        albums = response.json()
        for album in albums:
            if album.get('albumName') == album_name:
                return album.get('id')
        print(f"Warning: Album '{album_name}' not found.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting albums from Immich API: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding album list JSON: {response.text}")
        return None

def add_asset_to_immich_album(api_url: str, api_key: str, asset_id: str, album_id: str) -> bool:
    """Adds an asset to a specific Immich album."""
    endpoint = f"{api_url.rstrip('/')}/albums/{album_id}/assets"
    headers = {'x-api-key': api_key, 'Content-Type': 'application/json'}
    payload = {"ids": [asset_id]}
    try:
        response = requests.put(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        # Check response content for success indication if necessary
        # print(f"DEBUG API Response (Add to Album): {response.json()}")
        result = response.json()
        # Sample success: [{'id': 'album_id', 'success': True, 'error': None}]
        # Sample fail (already in album): [{'id': 'album_id', 'success': False, 'error': 'already_exists'}]
        if isinstance(result, list) and result and result[0].get('success') is True:
             print(f"  Successfully added asset {asset_id} to album {album_id}.")
             return True
        elif isinstance(result, list) and result and result[0].get('error') == 'already_exists':
             print(f"  Asset {asset_id} already in album {album_id}.")
             return True # Treat as success for our purpose
        elif isinstance(result, list) and result and result[0].get('error') == 'duplicate': # Handle duplicate error
             print(f"  Asset {asset_id} already in album {album_id} (reported as duplicate).")
             return True # Treat as success for our purpose
        else:
             print(f"  Failed to add asset {asset_id} to album {album_id}. Response: {result}")
             return False
    except requests.exceptions.RequestException as e:
        print(f"Error adding asset {asset_id} to album {album_id}: {e}")
        return False
    except json.JSONDecodeError:
         print(f"Error decoding add asset response JSON: {response.text}")
         return False

def visibility_immich_asset(api_url: str, api_key: str, asset_id: str, action: str) -> bool:
    """Updates asset visibility in Immich by updating their properties.
    
    Args:
        api_url: The Immich API URL
        api_key: The Immich API key
        asset_id: The ID of the asset to update
        action: The visibility action to take (archive, timeline, hidden, locked)
    
    Returns:
        bool: True if the update was successful, False otherwise
    """
    endpoint = f"{api_url.rstrip('/')}/assets"
    headers = {'x-api-key': api_key, 'Content-Type': 'application/json'}
    payload = {
        "ids": [asset_id],
        "visibility": action
    }
    try:
        response = requests.put(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        print(f"  Successfully updated visibility for asset {asset_id} to {action}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error updating visibility for asset {asset_id}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during visibility update for {asset_id}: {e}")
        return False

def get_immich_asset_details(api_url: str, api_key: str, asset_id: str) -> dict | None:
    """Fetches details for a specific asset from Immich."""
    endpoint = f"{api_url.rstrip('/')}/assets/{asset_id}"
    headers = {'x-api-key': api_key}
    try:
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting details for asset {asset_id}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding asset details JSON for {asset_id}: {response.text}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred fetching details for {asset_id}: {e}")
         return None

# ------------------------

# TODO: Add main logic

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Automatically classify Immich assets using CLIP.")
    parser.add_argument("--full-scan", action="store_true", 
                        help="Perform a full scan of all assets, ignoring timestamps and previously processed IDs.")
    parser.add_argument("--full-unarchived-scan", action="store_true", 
                        help="Perform a full scan of all unarchived assets, ignoring timestamps and previously processed IDs.")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()

    # Get config from environment
    immich_api_url = os.getenv("IMMICH_API_URL")
    immich_api_key = os.getenv("IMMICH_API_KEY")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    ml_api_url = os.getenv("ML_API_URL")
    clip_model_name = os.getenv("CLIP_MODEL_NAME")
    fetch_hours_str = os.getenv("FETCH_HOURS", "24") # Default to 24 if not set

    # Basic validation
    required_vars = {
        "IMMICH_API_URL": immich_api_url,
        "IMMICH_API_KEY": immich_api_key,
        "DB_NAME": db_name,
        "DB_USER": db_user,
        "DB_HOST": db_host,
        "DB_PORT": db_port,
        "ML_API_URL": ml_api_url,
        "CLIP_MODEL_NAME": clip_model_name,
        # FETCH_HOURS is optional with a default, but we should validate it if present
    }

    missing_vars = [k for k, v in required_vars.items() if v is None]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)

    # Validate FETCH_HOURS
    try:
        fetch_hours = int(fetch_hours_str)
        if fetch_hours <= 0:
            raise ValueError("FETCH_HOURS must be positive")
    except ValueError:
        print(f"Error: Invalid FETCH_HOURS value '{fetch_hours_str}'. Must be a positive integer.")
        exit(1)

    print("Configuration loaded successfully.")

    # --- DB Connections & Asset Fetching (Outer Try/Finally) ---
    conn = None
    cur = None
    sqlite_conn = None
    sqlite_cur = None
    new_assets_to_process = [] 
    processed_asset_ids = set() 

    try: # Outer try for DB connections and initial fetch
        # --- Initialize SQLite DB for Processed IDs ---
        sqlite_db_file = "processed_ids.sqlite"
        sqlite_conn = sqlite3.connect(sqlite_db_file)
        sqlite_cur = sqlite_conn.cursor()
        
        # Create table if it doesn't exist
        sqlite_cur.execute("""
            CREATE TABLE IF NOT EXISTS processed_assets (
                asset_id TEXT,
                matched_keyword TEXT,
                similarity_score REAL,
                target_album_id TEXT,
                target_album_name TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        sqlite_conn.commit()
        print(f"Using SQLite database: {sqlite_db_file}")

        if args.full_scan:
            print("*** Performing Full Scan - Ignoring previously processed IDs and timestamps ***")
            processed_asset_ids = set() # Start with an empty set for full scan
        elif args.full_unarchived_scan:
            # Load existing processed IDs from SQLite (only if NOT full scan)
            sqlite_cur.execute("SELECT asset_id FROM processed_assets")
            processed_asset_ids = set(row[0] for row in sqlite_cur.fetchall())
            print(f"Loaded {len(processed_asset_ids)} processed asset IDs from SQLite.")
        else:
            # Load existing processed IDs from SQLite (only if NOT full scan)
            sqlite_cur.execute("SELECT asset_id FROM processed_assets")
            processed_asset_ids = set(row[0] for row in sqlite_cur.fetchall())
            print(f"Loaded {len(processed_asset_ids)} processed asset IDs from SQLite.")
        # ----------------------------------------------
        
        # --- PostgreSQL Connection & Fetch ---
        # Option B: Attempt connection via Unix domain socket (like psql does)
        # This might work with peer authentication if run as the 'immich' OS user
        # Requires DB_USER='immich' in .env, DB_PASSWORD might be irrelevant/empty
        # Common socket directories: /run/postgresql, /var/run/postgresql, /tmp
        # psycopg2 might automatically find it if host/port are omitted
        print("Attempting database connection via Unix socket...")
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            # password=db_password, # Comment out or remove password for peer auth trial
            # host=db_host, # Omit host to encourage socket connection
            # port=db_port # Omit port
        )
        cur = conn.cursor()
        print("Database connection successful (via socket).")

        # Fetch assets from DB
        # Use timezone-aware UTC now
        # Use fetch_hours from config
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=fetch_hours)
        if not args.full_scan and not args.full_unarchived_scan:
             print(f"Fetching assets newer than {cutoff_time} ({fetch_hours} hours ago)...")
        recent_assets = []
        
        base_query = """
        SELECT a.id, ss."embedding"
        FROM assets a
        JOIN smart_search ss ON a.id = ss."assetId"
        """
        params = []
        
        if not args.full_scan and not args.full_unarchived_scan:
            query = base_query + " WHERE a.\"createdAt\" >= %s ORDER BY a.\"createdAt\" DESC;"
            params.append(cutoff_time)
        else:
            query = base_query + " ORDER BY a.\"createdAt\" DESC;" # Full scan, no time limit, no type filter
        
        # Important: Using parameterized queries to prevent SQL injection
        cur.execute(query, tuple(params))
        raw_assets = cur.fetchall()

        # Process raw results (assuming embedding is directly usable or needs simple handling)
        # We might need to adjust how 'embedding' is handled based on its actual type/format
        for i, (asset_id, embedding_raw) in enumerate(raw_assets):
            # Process the embedding string using json.loads
            try:
                # Attempt parsing the string and converting to numpy array
                embedding_list = json.loads(embedding_raw)
                asset_embedding = np.array(embedding_list, dtype=np.float32)

                # Check if conversion was successful (optional but good practice)
                if asset_embedding.ndim != 1 or asset_embedding.size == 0:
                     print(f"Warning: Embedding for asset {asset_id} resulted in unexpected shape {asset_embedding.shape}. Skipping.")
                     continue # Skip this asset

                recent_assets.append({"id": asset_id, "embedding": asset_embedding})

            except json.JSONDecodeError as json_e:
                print(f"Warning: Could not parse embedding string for asset {asset_id}: {json_e}. Skipping.")
                continue # Skip this asset
            except Exception as conversion_e:
                print(f"Warning: Could not process embedding for asset {asset_id}: {conversion_e}. Skipping.")
                continue # Skip this asset

        print(f"Fetched and processed {len(recent_assets)} image assets from the database.")

        # Filter out already processed assets (Skip if full_scan)
        if not args.full_scan:
            assets_before_filtering = len(recent_assets)
            new_assets_to_process = [
                asset for asset in recent_assets 
                if asset["id"] not in processed_asset_ids
            ]
            filtered_count = assets_before_filtering - len(new_assets_to_process)
            if filtered_count > 0:
                 print(f"Filtered out {filtered_count} previously processed assets.")
        else:
             # For full scan, process everything fetched
             new_assets_to_process = recent_assets 
             
        print(f"Found {len(new_assets_to_process)} assets to classify.")
        # ------------------------------------
        
        # === Main Processing (Inner Try) ===
        try:
            # If no new assets, exit (outer finally block will handle cleanup)
            if not new_assets_to_process:
                print("No new assets found requiring classification.")
                exit(0)

            # Load classification rules from JSON file
            config_file = "config.json"
            classification_rules = []
            classification_groups = [] # New: Load groups
            all_keywords = set()       # New: Collect all unique keywords
            
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    classification_rules = config_data.get("classification_rules", [])
                    classification_groups = config_data.get("classification_groups", []) # Load groups

                # --- Validate Individual Rules & Collect Keywords ---
                if not classification_rules:
                     print(f"Warning: No individual classification rules found or loaded from {config_file}.")
                else:
                     print(f"Loaded {len(classification_rules)} individual classification rules from {config_file}.")
                     for i, rule in enumerate(classification_rules):
                         if not all(k in rule for k in ["keyword", "min_similarity", "album_names"]):
                             print(f"Error: Rule {i} in {config_file} is missing required keys (keyword, min_similarity, album_names). Exiting.")
                             exit(1)
                         if not isinstance(rule['album_names'], list) or not rule['album_names']:
                              print(f"Error: Rule {i} 'album_names' must be a non-empty list. Exiting.")
                              exit(1)
                         rule['action'] = rule.get('action') 
                         if rule['action'] == "None": rule['action'] = None
                         all_keywords.add(rule["keyword"]) # Collect keyword

                # --- Validate Group Rules & Collect Keywords ---
                if not classification_groups:
                     print(f"Warning: No classification groups found or loaded from {config_file}.")
                else:
                    print(f"Loaded {len(classification_groups)} classification groups from {config_file}.")
                    for i, group in enumerate(classification_groups):
                         if not all(k in group for k in ["group_name", "keywords", "min_sum_similarity", "album_names"]):
                             print(f"Error: Group {i} in {config_file} is missing required keys (group_name, keywords, min_sum_similarity, album_names). Exiting.")
                             exit(1)
                         if not isinstance(group['keywords'], list) or not group['keywords']:
                             print(f"Error: Group {i} 'keywords' must be a non-empty list. Exiting.")
                             exit(1)
                         group['action'] = group.get('action')
                         if group['action'] == "None": group['action'] = None
                         for kw in group["keywords"]: # Collect keywords from group
                             all_keywords.add(kw)
                             
            except FileNotFoundError:
                 print(f"Error: Configuration file {config_file} not found. Exiting.")
                 exit(1)
            except json.JSONDecodeError as e:
                 print(f"Error decoding JSON from {config_file}: {e}. Exiting.")
                 exit(1)
            except Exception as e:
                print(f"An unexpected error occurred loading config: {e}. Exiting.")
                exit(1)

            # Exit if no rules OR groups were loaded
            if not classification_rules and not classification_groups:
                print("No classification rules or groups found to process. Exiting.")
                exit(0)

            # Get text embeddings for ALL unique keywords
            if not all_keywords:
                 print("No keywords found in rules or groups. Exiting.")
                 exit(0)
                 
            print(f"\nFetching text embeddings for {len(all_keywords)} unique keywords...")
            keyword_embeddings_map = get_text_embeddings(list(all_keywords), ml_api_url, clip_model_name)

            if keyword_embeddings_map is None:
                print("Failed to fetch keyword embeddings. Exiting.")
                exit(1)
            
            # --- Process Assets ---
            print(f"\nProcessing {len(new_assets_to_process)} assets...")
            actions_to_perform = [] # Reset actions list for this run
            matched_asset_ids = set() # Track assets that had at least one match (either rule or group)

            for asset in new_assets_to_process:
                asset_id = asset["id"]
                image_embedding = asset["embedding"]
                asset_matched_something = False # Flag for this specific asset (rule or group)
                
                # --- 1. Check against Individual Rules --- 
                for rule in classification_rules:
                    keyword = rule["keyword"]
                    keyword_embedding = keyword_embeddings_map.get(keyword)
                    
                    if keyword_embedding is None:
                         # This shouldn't happen if get_text_embeddings worked, but check defensively
                         print(f"Warning: Embedding missing for keyword '{keyword}' during rule check. Skipping rule.")
                         continue 
                         
                    similarity = cosine_similarity(image_embedding, keyword_embedding)

                    # Check if THIS rule's threshold is met
                    if similarity >= rule["min_similarity"]:
                        asset_matched_something = True
                        matched_asset_ids.add(asset_id) # Add to set of matched assets
                        print(f"  Asset {asset_id}: [Rule Match] Keyword: '{rule['keyword']}' (Sim: {similarity:.3f} >= {rule['min_similarity']:.3f}) -> Albums: {rule['album_names']}, Action: {rule['action']}")
                        # Store action details for this specific match
                        actions_to_perform.append({
                            "asset_id": asset_id,
                            "album_names": rule["album_names"],
                            "action_type": rule.get("visibility"),  # Use visibility field instead of action
                            "matched_keyword": rule["keyword"], # Log the specific keyword
                            "similarity_score": similarity     # Log the individual similarity
                        })
                        # Note: We trigger for ALL individual matches above threshold

                # --- 2. Check against Group Rules ---
                for group in classification_groups:
                    group_name = group["group_name"]
                    group_keywords = group["keywords"]
                    min_sum_similarity = group["min_sum_similarity"]
                    total_similarity = 0.0
                    keywords_in_sum = [] # Track keywords contributing to the sum for this group

                    for keyword in group_keywords:
                        keyword_embedding = keyword_embeddings_map.get(keyword)
                        if keyword_embedding is None:
                            print(f"Warning: Embedding missing for keyword '{keyword}' in group '{group_name}'. Skipping keyword.")
                            continue
                            
                        similarity = cosine_similarity(image_embedding, keyword_embedding)
                        total_similarity += similarity
                        keywords_in_sum.append(f"'{keyword}' ({similarity:.3f})") # For logging

                    # Check if THIS group's threshold is met
                    if total_similarity >= min_sum_similarity:
                         asset_matched_something = True
                         matched_asset_ids.add(asset_id) # Add to set of matched assets
                         print(f"  Asset {asset_id}: [Group Match] Group: '{group_name}' (Sum: {total_similarity:.3f} >= {min_sum_similarity:.3f}) (From: {', '.join(keywords_in_sum)}) -> Albums: {group['album_names']}, Action: {group['action']}")
                         # Store action details for this group match
                         actions_to_perform.append({
                             "asset_id": asset_id,
                             "album_names": group["album_names"],
                             "action_type": group.get("visibility"),  # Use visibility field instead of action
                             "matched_keyword": f"Group: {group_name}", # Log the group name
                             "similarity_score": total_similarity     # Log the summed similarity
                         })
                         # Note: We trigger for ALL group matches above threshold
                
                # Optional: Log if an asset didn't match ANYTHING (neither rule nor group)
                if not asset_matched_something:
                    print(f"  Asset {asset_id}: No rules or groups met threshold. Skipping.")

            print("\nAsset classification complete.")
            # ------------------------

            # --- Perform Immich API Actions --- 
            print(f"\nPerforming actions for {len(matched_asset_ids)} matched assets via Immich API...")
            
            # Get unique album names needed for this run
            album_names_needed = set()
            for action in actions_to_perform:
                for name in action['album_names']:
                    album_names_needed.add(name)
            album_id_cache = {}
            
            # Fetch and cache album IDs
            if album_names_needed:
                print("Fetching album IDs...")
                all_albums_found = True
                for name in sorted(list(album_names_needed)): # Sort for consistent log order
                    album_id = get_immich_album_id(immich_api_url, immich_api_key, name)
                    if album_id:
                        album_id_cache[name] = album_id
                        print(f"  Found Album '{name}': ID {album_id}")
                    else:
                        print(f"  Error: Could not find Album ID for '{name}'. Actions for this album will be skipped.")
                        all_albums_found = False
                # Optional: exit if any albums are missing? -> Kept commented out as before

            # Group actions by asset_id to handle archiving correctly
            actions_by_asset = {}
            for action in actions_to_perform:
                asset_id = action['asset_id']
                if asset_id not in actions_by_asset:
                    actions_by_asset[asset_id] = []
                actions_by_asset[asset_id].append(action)
                
            print(f"Processing actions for {len(actions_by_asset)} unique assets that matched rules or groups...")
            
            # Perform actions for each matched asset
            successful_album_adds = 0 
            db_log_entries_queued = 0 # Counter for queued logs
            unique_album_adds_for_asset = set() # Track unique album additions per asset to avoid duplicates
            should_update_visibility = True # Flag to control visibility updates
            for asset_id, asset_actions in actions_by_asset.items():
                print(f"-- Processing Asset {asset_id} --")
                added_to_at_least_one_album = False  # Initialize to False
                
                # Only try to add to albums if there are any album names
                if any(action["album_names"] for action in asset_actions):
                    for action in asset_actions:
                        album_names = action["album_names"]
                        # Use the specific keyword or group name stored in the action for logging
                        trigger_source = action["matched_keyword"]
                        score = action["similarity_score"]
                        
                        for album_name in album_names:
                            album_id = album_id_cache.get(album_name)
                            if not album_id:
                                # This message is fine as is
                                print(f"  Skipping add to album '{album_name}' - Album ID not found.")
                                continue
                                
                            # Avoid redundant API calls if multiple rules/groups point to same album for same asset
                            log_key = (asset_id, album_id)
                            if log_key in unique_album_adds_for_asset:
                                continue 

                            print(f"  Attempting to add asset {asset_id} to album '{album_name}' (ID: {album_id}) triggered by '{trigger_source}'...") 
                            added_to_this_album = add_asset_to_immich_album(immich_api_url, immich_api_key, asset_id, album_id)
                            unique_album_adds_for_asset.add(log_key) # Mark as attempted
                            
                            if added_to_this_album:
                                successful_album_adds += 1
                                added_to_at_least_one_album = True
                                # --- IMMEDIATE DB LOG --- 
                                db_log_entry = (asset_id, trigger_source, float(score), album_id, album_name)
                                try:
                                    if sqlite_conn and sqlite_cur:
                                        sqlite_cur.execute("""
                                            INSERT OR IGNORE INTO processed_assets 
                                            (asset_id, matched_keyword, similarity_score, target_album_id, target_album_name)
                                            VALUES (?, ?, ?, ?, ?)
                                            """, db_log_entry)
                                        db_log_entries_queued += 1 # Increment counter
                                    else:
                                        print("    -> DB connection not available for queueing log.")
                                except sqlite3.Error as e:
                                    print(f"    -> ERROR queueing DB log entry: {e}")
                                # --- END IMMEDIATE DB LOG ---
                else:
                    print(f"  No album names specified for asset {asset_id}, skipping album additions.")
                
                # --- Step 2: Update visibility if needed and possible --- 
                # Now we check for visibility updates regardless of album additions
                if should_update_visibility:
                    asset_details = get_immich_asset_details(immich_api_url, immich_api_key, asset_id)
                    # Get the requested visibility from the first matching rule/group that has a non-empty action_type
                    requested_visibility = next((a['action_type'] for a in asset_actions if a['action_type'] and a['action_type'] != 'None'), None)
                    
                    if requested_visibility and asset_details and asset_details.get('visibility') != requested_visibility:
                        print(f"  Asset {asset_id} visibility is {asset_details.get('visibility')}. Attempting to set visibility to {requested_visibility} (requested by at least one rule/group)...")
                        visibility_success = visibility_immich_asset(immich_api_url, immich_api_key, asset_id, requested_visibility)
                        if not visibility_success:
                            print(f"Warning: Failed to update visibility for asset {asset_id} after adding to album(s).")
                    elif asset_details and asset_details.get('visibility') == requested_visibility:
                        print(f"  Skipping visibility update for asset {asset_id}: Already has requested visibility {requested_visibility}.")
                    elif not requested_visibility:
                        print(f"  Skipping visibility update for asset {asset_id}: No valid visibility action requested.")
                    else:
                        print(f"Warning: Could not determine visibility status for asset {asset_id}. Skipping visibility update attempt.")
                else:
                    print(f"  Skipping visibility update for asset {asset_id}: Not requested by any matching rule or group.")
            # --- End Asset Processing Loop --- 
             
            print(f"\nImmich API actions complete. {successful_album_adds} total successful album additions performed." )        
            # --------------------------------

            # --- Update Processed IDs DB ---
            # Now we just need to commit the changes queued earlier
            print(f"\nCommitting {db_log_entries_queued} logged entries to processed IDs database...")
            if sqlite_conn:
                try:
                    # Check count before commit (optional)
                    # count_before = 0
                    # if sqlite_cur:
                    #     try:
                    #         sqlite_cur.execute("SELECT COUNT(*) FROM processed_assets")
                    #         count_before = sqlite_cur.fetchone()[0]
                    #     except sqlite3.Error as e_count:
                    #         print(f"Warning: Could not get count before commit: {e_count}")
                            
                    sqlite_conn.commit()
                    
                    # Query count after commit for confirmation
                    count_after = -1
                    if sqlite_cur:
                         try:
                             sqlite_cur.execute("SELECT COUNT(*) FROM processed_assets")
                             count_after = sqlite_cur.fetchone()[0]
                             print(f"Successfully committed changes. Total entries in processed_assets now: {count_after}")
                         except sqlite3.Error as e_count:
                              print(f"Successfully committed changes, but failed to get count after: {e_count}")
                    else:
                         print("Successfully committed changes (cursor unavailable for count confirmation).")

                except sqlite3.Error as e:
                    print(f"Error committing to SQLite processed IDs database: {e}")
                    # Rollback might still be useful if commit failed mid-way
                    try:
                        print("Attempting to rollback transaction...")
                        sqlite_conn.rollback()
                        print("Rolled back transaction due to commit error.")
                    except sqlite3.Error as rb_e:
                        print(f"Error during rollback after commit error: {rb_e}")
            elif db_log_entries_queued > 0:
                 print("Skipping commit because SQLite connection is not available, but entries were queued.")
            else:
                 print("No new entries were queued, skipping commit.")
            # --------------------------------
        
        except Exception as e_main: # Catch unexpected errors during main processing
            print(f"An unexpected error occurred during main processing: {e_main}")
            # Log or handle specific errors as needed
        # === End Inner Main Processing Try ===

    except psycopg2.Error as e_pg:
        print(f"PostgreSQL query error: {e_pg}")
        if conn: conn.rollback() 
        # Exit here, finally will close connections opened before the error
        exit(1) 
    except sqlite3.Error as e_sqlite:
         print(f"SQLite setup/load error: {e_sqlite}")
         # Exit here, finally will close connections opened before the error
         exit(1)
    except Exception as e_setup:
        print(f"An unexpected error occurred during DB setup or initial fetch: {e_setup}")
        if conn: conn.rollback() # Rollback PG if connection was made
        # Exit here, finally will close connections opened before the error
        exit(1)
    # --- End Outer DB Setup/Fetch Try/Except Block ---
    
    finally: # Associated with the OUTER try block
        # --- Final Database Cleanup --- 
        print("\nClosing database connections...")
        # Close PostgreSQL connection
        if cur:
            try: cur.close()
            except Exception as e: print(f"Ignoring error closing PG cursor: {e}")
        if conn:
            try: conn.close(); print("PostgreSQL connection closed.")
            except Exception as e: print(f"Ignoring error closing PG connection: {e}")
        # Close SQLite connection
        if sqlite_cur:
             try: sqlite_cur.close()
             except Exception as e: print(f"Ignoring error closing SQLite cursor: {e}")
        if sqlite_conn:
             try: sqlite_conn.close(); print("SQLite connection closed.")
             except Exception as e: print(f"Ignoring error closing SQLite connection: {e}")
    # === End Outer Try/Finally ===

# ------------------------ 
