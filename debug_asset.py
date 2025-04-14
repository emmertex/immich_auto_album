import os
import psycopg2
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import json
import time
import sqlite3 # Although not used for writing, keep for potential future debug needs
import argparse

# --- Helper Functions (Copied from main.py) ---

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
            "entries": (None, json.dumps(request_config)),
            "text": (None, text_input)
        }
        print(f"  Requesting embedding for: '{text_input}'")
        try:
            response = requests.post(endpoint, files=form_data)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "clip" in data and isinstance(data["clip"], str):
                embedding_string = data["clip"]
                embedding_list = json.loads(embedding_string)
            else:
                print(f"Error: Unexpected response format from ML predict for '{text_input}': {data}")
                # Continue trying other keywords? For debug, maybe best to fail hard.
                return None

            embedding = np.array(embedding_list, dtype=np.float32)
            expected_dim = 1024 # Assuming same dimension as main.py
            if embedding.shape[0] != expected_dim:
                 print(f"Warning: ML service returned embedding with unexpected shape (Expected {expected_dim}) for '{text_input}': {embedding.shape}")
                 # Decide if this is fatal for debug script

            embeddings_map[text_input] = embedding
            # time.sleep(0.1) # Optional delay

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
         print("Error: Failed to get embeddings for all requested texts.")
         return None

    print("Successfully fetched all keyword embeddings.")
    return embeddings_map

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates the cosine similarity between two numpy vectors."""
    if vec1.shape != vec2.shape:
        print(f"Warning: Shape mismatch for similarity calc: {vec1.shape} vs {vec2.shape}")
        return 0.0
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

# --- Main Debug Logic ---

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Debug classification logic for a single Immich asset.")
    parser.add_argument("asset_id", help="The ID of the asset to debug.")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get config from environment (similar to main.py)
    immich_api_url = os.getenv("IMMICH_API_URL")
    immich_api_key = os.getenv("IMMICH_API_KEY")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    ml_api_url = os.getenv("ML_API_URL")
    clip_model_name = os.getenv("CLIP_MODEL_NAME")

    # Basic validation
    required_vars = {
        "IMMICH_API_URL": immich_api_url, # Needed for potential future API calls in debug? Keep for now.
        "IMMICH_API_KEY": immich_api_key,
        "DB_NAME": db_name,
        "DB_USER": db_user,
        "DB_HOST": db_host,
        "DB_PORT": db_port,
        "ML_API_URL": ml_api_url,
        "CLIP_MODEL_NAME": clip_model_name,
    }
    missing_vars = [k for k, v in required_vars.items() if v is None]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)

    print("Configuration loaded successfully.")
    print(f"Debugging asset ID: {args.asset_id}")

    # --- DB Connection & Asset Fetch ---
    conn = None
    cur = None
    asset_embedding = None

    try:
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user
        )
        cur = conn.cursor()
        print("Database connection successful.")

        # Fetch the specific asset's embedding
        query = """
        SELECT ss."embedding"
        FROM smart_search ss
        WHERE ss."assetId" = %s;
        """
        cur.execute(query, (args.asset_id,))
        result = cur.fetchone()

        if not result:
            print(f"Error: Asset ID {args.asset_id} not found in smart_search table.")
            exit(1)

        embedding_raw = result[0]
        try:
            embedding_list = json.loads(embedding_raw)
            asset_embedding = np.array(embedding_list, dtype=np.float32)
            print(f"Successfully fetched embedding for asset {args.asset_id} (Shape: {asset_embedding.shape})")
        except json.JSONDecodeError as json_e:
            print(f"Error: Could not parse embedding string for asset {args.asset_id}: {json_e}.")
            exit(1)
        except Exception as conversion_e:
            print(f"Error: Could not process embedding for asset {args.asset_id}: {conversion_e}.")
            exit(1)

    except psycopg2.Error as e_pg:
        print(f"PostgreSQL connection or query error: {e_pg}")
        exit(1)
    except Exception as e_setup:
        print(f"An unexpected error occurred during DB connection or fetch: {e_setup}")
        exit(1)
    finally:
        if cur: cur.close()
        if conn: conn.close()
        print("Database connection closed.")

    if asset_embedding is None:
        print("Failed to retrieve asset embedding. Exiting.")
        exit(1)

    # --- Load Configuration and Keywords ---
    config_file = "config.json"
    classification_rules = []
    classification_groups = []
    all_keywords = set()

    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            classification_rules = config_data.get("classification_rules", [])
            classification_groups = config_data.get("classification_groups", []) # Load groups

            # Validate and collect keywords from individual rules
            print(f"Loaded {len(classification_rules)} individual classification rules.")
            for i, rule in enumerate(classification_rules):
                if not all(k in rule for k in ["keyword", "min_similarity", "album_names"]):
                    print(f"Error: Rule {i} in {config_file} is missing required keys (keyword, min_similarity, album_names). Exiting.")
                    exit(1)
                rule['action'] = rule.get('action') # Ensure action is present (can be None)
                if rule['action'] == "None": rule['action'] = None
                all_keywords.add(rule["keyword"])

            # Validate and collect keywords from groups
            print(f"Loaded {len(classification_groups)} classification groups.")
            for i, group in enumerate(classification_groups):
                 if not all(k in group for k in ["group_name", "keywords", "min_sum_similarity", "album_names"]):
                     print(f"Error: Group {i} in {config_file} is missing required keys (group_name, keywords, min_sum_similarity, album_names). Exiting.")
                     exit(1)
                 if not isinstance(group['keywords'], list) or not group['keywords']:
                     print(f"Error: Group {i} 'keywords' must be a non-empty list. Exiting.")
                     exit(1)
                 group['action'] = group.get('action') # Ensure action is present (can be None)
                 if group['action'] == "None": group['action'] = None
                 for kw in group["keywords"]:
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

    if not classification_rules and not classification_groups:
        print("No classification rules or groups found in config. Exiting.")
        exit(0)

    # --- Get Keyword Embeddings ---
    print(f"Fetching embeddings for {len(all_keywords)} unique keywords...")
    keyword_embeddings_map = get_text_embeddings(list(all_keywords), ml_api_url, clip_model_name)

    if keyword_embeddings_map is None:
        print("Failed to fetch keyword embeddings. Exiting.")
        exit(1)

    # --- Perform Classification Analysis ---
    print(f"--- Analyzing Asset {args.asset_id} ---")

    # 1. Individual Rule Matching
    print("Individual Rule Matches:")
    matched_individual_rules = False
    for rule in classification_rules:
        keyword = rule["keyword"]
        keyword_embedding = keyword_embeddings_map.get(keyword)
        if keyword_embedding is None:
            print(f"  Warning: Embedding not found for keyword '{keyword}'. Skipping rule.")
            continue

        similarity = cosine_similarity(asset_embedding, keyword_embedding)
        meets_threshold = similarity >= rule["min_similarity"]
        if meets_threshold:
            matched_individual_rules = True
            print(f"  [MATCH] Keyword: '{keyword}'")
            print(f"    Similarity: {similarity:.4f} >= Threshold: {rule['min_similarity']:.4f}")
            print(f"    Albums: {rule['album_names']}, Action: {rule['action']}")
        else:
            print(f"  [NO MATCH] Keyword: '{keyword}'")
            print(f"    Similarity: {similarity:.4f} < Threshold: {rule['min_similarity']:.4f}")

    if not matched_individual_rules:
        print("  No individual rules met their thresholds.")

    # 2. Group Rule Matching
    print("Group Rule Matches:")
    matched_group_rules = False
    for group in classification_groups:
        group_name = group["group_name"]
        group_keywords = group["keywords"]
        min_sum_similarity = group["min_sum_similarity"]
        total_similarity = 0.0
        print(f"Group: '{group_name}' (Threshold Sum: {min_sum_similarity:.4f})")

        for keyword in group_keywords:
            keyword_embedding = keyword_embeddings_map.get(keyword)
            if keyword_embedding is None:
                print(f"    Warning: Embedding not found for keyword '{keyword}' within group. Skipping keyword.")
                continue

            similarity = cosine_similarity(asset_embedding, keyword_embedding)
            print(f"    Keyword: '{keyword}', Similarity: {similarity:.4f}")
            total_similarity += similarity

        meets_threshold = total_similarity >= min_sum_similarity
        print(f"    Calculated Sum: {total_similarity:.4f}")
        if meets_threshold:
            matched_group_rules = True
            print(f"  [GROUP MATCH] Group '{group_name}' threshold met!")
            print(f"    Sum: {total_similarity:.4f} >= Threshold Sum: {min_sum_similarity:.4f}")
            print(f"    Albums: {group['album_names']}, Action: {group['action']}")
        else:
            print(f"  [NO GROUP MATCH] Group '{group_name}' threshold not met.")
            print(f"    Sum: {total_similarity:.4f} < Threshold Sum: {min_sum_similarity:.4f}")

    if not matched_group_rules:
        print("  No groups met their sum thresholds.")

    print("--- Debug Analysis Complete ---")
