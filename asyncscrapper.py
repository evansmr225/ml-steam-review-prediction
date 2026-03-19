import asyncio
import aiohttp
import pandas as pd
import os
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from dateutil import parser
import requests
import random
import numpy as np
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("STEAM_API_KEY")
CSV_FILE = "steam_raw_data.csv"
FINAL_CSV = "steam_ml_cleaned_data.csv"
CONCURRENT_REQUESTS = 2  
SAVE_EVERY = 100
LOG_FILE = "crawl_heartbeat.txt"         

# --- DATA COLLECTION FUNCTIONS ---
def get_all_game_ids(api_key, target=50000):
    """
    Retrieves a comprehensive list of Game AppIDs using cursor-based pagination.
    
    Filters for standalone games only, excluding DLC, Soundtracks, and Software.
    Steam's API limits single requests to 50k results, so this loops using 
    'last_appid' until the entire catalog is retrieved.
    """
    all_ids = []
    last_appid = 0
    url = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
    
    print("Beginning Steam Catalog Scan...")

    while True:
        params = {
            "key": api_key,
            "include_games": True,
            "include_dlc": False,
            "include_software": False,
            "include_video": False,
            "include_hardware": False,
            "max_results": target,
            "last_appid": last_appid  
        }
        
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            
            response_data = res.json().get("response", {})
            apps = response_data.get("apps", [])
            
            if not apps:
                break 
                
            # extract AppIDs and append to our master list
            batch_ids = [app['appid'] for app in apps]
            all_ids.extend(batch_ids)
            
            # update the cursor for the next iteration
            last_appid = response_data.get("last_appid")
            
            print(f"Total IDs Collected: {len(all_ids)} | Current Cursor: {last_appid}")
            
            # steam explicitly signals if more data exists via 'have_more_results'
            if not response_data.get("have_more_results", False):
                break

        except Exception as e:
            print(f"Pagination interrupted at AppID {last_appid}: {e}")
            break

    print(f"Retrieval Complete. Final Count: {len(all_ids)} Games.")
    return all_ids

def get_gpu_score(reqs_html):
    """
    Parses the system requirements HTML to estimate GPU power needs.
    0: Low/Unknown, 1: Integrated/Entry, 2: Mid-range, 3: High-end (RTX/High VRAM)
    """
    if not reqs_html:
        return 0
    text = reqs_html.lower()
    
    # Cceck for high-end markers
    if any(x in text for x in ["rtx", "3080", "4070", "4080", "4090", "8gb vram", "12gb vram"]):
        return 3
    # check for mid-range
    if any(x in text for x in ["gtx", "1060", "2060", "4gb vram", "6gb vram"]):
        return 2
    # check for entry-level
    if any(x in text for x in ["mb video memory", "shader model", "intel hd", "integrated", "minimum", "directx 9"]):
        return 1
    return 0

# --- ASYNC DATA COLLECTION ---

async def fetch_game_snapshot(session, app_id, api_key, semaphore):
    """
    Gathers comprehensive Steam data with Exponential Backoff and 429 recovery.
    Retries the same ID until success or max_retries reached.
    """
    max_retries = 5
    retry_count = 0
    backoff_base = 15  
    
    # headers to mimic a real browser visit
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    while retry_count < max_retries:
        async with semaphore:
            try:
                # fetch store Details
                store_url = "https://store.steampowered.com/api/appdetails"
                async with session.get(store_url, params={"appids": app_id, "cc": "us"}, headers=headers, timeout=15) as resp:
                    # steam is made at us for going too fast and won't give us data
                    if resp.status == 429:
                        # we exponentially back off and retry
                        wait = (backoff_base * (2 ** retry_count)) + random.uniform(2, 7)
                        print(f"!!! [Rate Limit] AppID {app_id}. Waiting {int(wait)}s to retry...")
                        await asyncio.sleep(wait)
                        retry_count += 1
                        continue # Retry the loop for the same app_id
                    
                    # skip game for other HTTP errors
                    if resp.status != 200:
                        print(f"--- SKIPPING AppID {app_id}: Response status != 200. ---")
                        return None
                    
                    # get data as python dict
                    res = await resp.json()

                if not res or not res[str(app_id)]['success']:
                    print(f"--- SKIPPING AppID {app_id}: no success. ---")
                    return None
                
                data = res[str(app_id)]['data']
                if data.get("type") != "game":
                    print(f"--- SKIPPING AppID {app_id}: not a game. ---")
                    return None

                # fetch Review Stats
                review_url = f"https://store.steampowered.com/appreviews/{app_id}"
                async with session.get(review_url, params={"json": 1, "purchase_type": "all"}, headers=headers) as resp:
                    rev_res = await resp.json()
                rev_stats = rev_res.get("query_summary", {})

                # fetch Live Player Count
                players_url = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
                async with session.get(players_url, params={"key": api_key, "appid": app_id}, headers=headers) as resp:
                    p_res = await resp.json()
                current_players = p_res.get("response", {}).get("player_count", 0)

                # fetch SteamSpy Data for Playtime & Player Count
                spy_url = f"https://steamspy.com/api.php?request=appdetails&appid={app_id}"
                try:
                    async with session.get(spy_url, timeout=10) as resp:
                        if resp.status == 200:
                            spy_res = await resp.json()
                            avg_playtime_forever = spy_res.get("average_forever", 0)
                            avg_playtime_2weeks = spy_res.get("average_2weeks", 0)
                            peak_players_yesterday = spy_res.get("ccu", 0) 
                        else:
                            avg_playtime_forever, avg_playtime_2weeks, peak_players_yesterday = 0, 0, 0
                except Exception:
                    avg_playtime_forever, avg_playtime_2weeks, peak_players_yesterday = 0, 0, 0

                # --- PARSING & FEATURE ENGINEERING ---
                
                # Handling GPU Requirements Safely
                pc_reqs_data = data.get("pc_requirements", {})
                pc_reqs_html = pc_reqs_data.get("minimum", "") if isinstance(pc_reqs_data, dict) else ""
                gpu_power = get_gpu_score(pc_reqs_html)

                # Handling Price Safely
                price_info = data.get("price_overview", {})
                if price_info is None:
                    base_price, discount = 0.0, 0
                else:
                    base_price = price_info.get("initial", 0) / 100
                    discount = price_info.get("discount_percent", 0)

                # Organize final dictionary
                return {
                    "appid": app_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "name": data.get("name"),
                    "developer": "|".join(data.get("developers", [])),
                    "base_price": base_price,
                    "current_discount": discount,
                    "total_reviews": rev_stats.get("total_reviews", 0),
                    "positive_reviews": rev_stats.get("total_positive", 0),
                    "release_date": data.get("release_date", {}).get("date"),
                    "is_windows": int(data.get("platforms", {}).get("windows", False)),
                    "is_linux": int(data.get("platforms", {}).get("linux", False)),
                    "is_mac": int(data.get("platforms", {}).get("mac", False)),
                    "gpu_power_level": gpu_power,
                    "is_free": int(data.get("is_free", False)),
                    "genres": "|".join([g['description'] for g in data.get("genres", [])]),
                    "categories": "|".join([c['description'] for c in data.get("categories", [])]),
                    "num_dlcs": len(data.get("dlc", [])),
                    "current_player_count": current_players,
                    "peak_players_yesterday": peak_players_yesterday,
                    "average_playtime_forever": avg_playtime_forever,
                    "average_playtime_past_2_weeks": avg_playtime_2weeks,
                }

            except Exception as e:
                print(f"Error on AppID {app_id}: {str(e)}. Retrying in 10s...")
                await asyncio.sleep(10)
                retry_count += 1
                
    return None # if it fails all retries

async def main_crawl():
    """
    Gathers list of game ids to gather data from and filters them for ones not currently in csv.
    Asynchronously gathers data for games while saving records at a set rate
    """
    
    # get the target list of IDs
    all_ids = get_all_game_ids(API_KEY, target=100000)
    
    # check resume status
    processed_ids = set()
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        try:
            # We only read the appid column to save memory
            existing_df = pd.read_csv(CSV_FILE, usecols=['appid'])
            processed_ids = set(existing_df['appid'].unique())
            print(f"--- RESUME: Found {len(processed_ids)} games in CSV. Skipping... ---")
        except Exception as e:
            print(f"Could not read existing CSV ({e}). Starting fresh.")

    # filter IDs that haven't been processed yet
    ids_to_fetch = [aid for aid in all_ids if aid not in processed_ids]
    print(f"Total IDs to attempt: {len(ids_to_fetch)}")

    # initializes asynchronous semaphore object that acts as concurrency limiter
    # we probably should just set it to one to avoid rate limitng
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS) 
    
    async with aiohttp.ClientSession() as session:
        # holds async tasks for current batch
        tasks = []
        # holds successful results to be written to disk
        batch = []
        # tracks total number of games saved so far
        success_count = len(processed_ids)
        
        for i, aid in enumerate(ids_to_fetch):
            # Create the task
            tasks.append(fetch_game_snapshot(session, aid, API_KEY, semaphore))
            
            # Execute batch save every X games
            if len(tasks) >= SAVE_EVERY or i == len(ids_to_fetch) - 1:
                # gather runs the batch with exponential backoff inside each task
                results = await asyncio.gather(*tasks)
                
                # Filter out the Nones (failed games)
                valid_results = [r for r in results if r]
                batch.extend(valid_results)
                success_count += len(valid_results)
                
                # Physical Save to Disk
                if batch:
                    file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0
                    pd.DataFrame(batch).to_csv(
                        CSV_FILE, 
                        mode='a', 
                        header=not file_exists, 
                        index=False, 
                        encoding='utf-8-sig'
                    )

                    with open(LOG_FILE, "a") as f:
                        f.write(f"[{datetime.now()}] Success: {success_count} | Last ID: {batch[-1]['appid']}\n")
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Saved batch of {len(batch)}. Total games in CSV: {success_count}")
                    
                    batch = [] # Clear the list for the next batch
                
                tasks = [] # Clear the task list
                
                # Jittered sleep between batches to break the "bot pattern"
                wait_time = random.uniform(3.0, 7.0)
                await asyncio.sleep(wait_time)

    print(f"\nFinished! Total games collected: {success_count}")

# --- DATA PROCESSING FUNCTIONS ---

def get_top_n_binary_df(df, column_name, n=50, exclude_list=[]):
    """Splits by pipe and binarizes the top N tags."""
    # Convert "Action|Indie" -> ["Action", "Indie"]
    converted_lists = df[column_name].fillna("").str.split('|')
    
    # Count frequency
    all_items = [
        item for sublist in converted_lists 
        for item in sublist 
        if item and item not in exclude_list
    ]
    all_items = [item for sublist in converted_lists for item in sublist if item]
    top_items = [name for name, count in Counter(all_items).most_common(n)]
    
    print(f"Top {n} {column_name} selected: {top_items}")
    
    # Binarize
    mlb = MultiLabelBinarizer(classes=top_items)
    matrix = mlb.fit_transform(converted_lists)
    return pd.DataFrame(matrix, columns=mlb.classes_)

def get_historical_developer_prestige_metrics(df):
    """Calculates number of games primary developer previously released and their average positive review ratio"""

    # sort by developer and release date and create group for primary developer
    # release date is set to today if it is 'coming soon' string 
    # for purposes of this function we only care about games released by developer before game release date
    # which is all games
    df['release_date_helper'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_date_helper'] = df['release_date_helper'].fillna(pd.Timestamp(datetime.now().date()))
    df['primary_developer'] = df['developer'].str.split('|').str[0]
    df = df.sort_values(['primary_developer', 'release_date_helper']).reset_index(drop=True)
    group = df.groupby('primary_developer')

    # get the number of games released before this game and the mean 
    # of the positive reviews of the game
    df['dev_prev_game_count'] = group.cumcount() 
    df['dev_prev_avg_review_ratio'] = group['positive_review_ratio'].transform(lambda x: x.shift(1).expanding().mean())
    df['dev_prev_avg_review_ratio'] = df['dev_prev_avg_review_ratio'].fillna(0)

    return df.drop(['primary_developer', 'release_date_helper'], axis=1)

def process_release_date(df):
    """Creates days since release and release month (1-12) features from release date"""
    current_date = datetime.now()
    
    def parse_steam_date(date_str):
        try:
            # parser.parse handles almost any date format automatically
            release_date = parser.parse(str(date_str))
            
            days_since = (current_date - release_date).days
            month_of_release = release_date.month
            
            # Ensure we don't have negative days for future releases
            return max(0, days_since), month_of_release
        except:
            # Fallback if the date is "Coming Soon" or invalid
            return 0, 0

    # Apply the function and split into two columns
    df[['days_since_release', 'release_month']] = df['release_date'].apply(
        lambda x: pd.Series(parse_steam_date(x))
    )
    
    return df

def get_ratio_columns(df):
    """Creates percentage of positive reviews and engagement ratio"""

    df['positive_review_ratio'] = (df['positive_reviews'] / df['total_reviews']).fillna(0)
    
    # If forever > 0, do the math. Otherwise, result is 0.
    df['engagement_ratio'] = np.where(
        df['average_playtime_forever'] > 0, 
        df['average_playtime_past_2_weeks'] / df['average_playtime_forever'], 
        0
    )
    
    return df

def finalize_dataset(df):
    """Drops unecessary columns and reorders columns"""
    # Filter out titles with non ASCII characters
    # These are probably not real games
    if 'title' in df.columns:
        is_ascii = df['title'].fillna('').str.isascii()
        df = df[is_ascii].copy()

    # Drop unnecessary columns
    cols_to_drop = ['developer', 'release_date', 'positive_reviews']
    
    # Use errors='ignore' so the code doesn't crash if a column was already dropped
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Reorder columns so the label is last
    target_col = 'positive_review_ratio'
    
    if target_col in df.columns:
        # Create a list of all columns except the target
        other_cols = [c for c in df.columns if c != target_col]
        # Reconstruct the dataframe
        df = df[other_cols + [target_col]]
    
    df = df.sort_values(['appid'])
    return df

if __name__ == "__main__":
    # Run the Collector
    try:
        asyncio.run(main_crawl())
        print(f"\nCollection phase complete! Raw data at: {CSV_FILE}")
    except KeyboardInterrupt:
        print("\nCrawl paused by user. You can restart later to resume.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        with open(LOG_FILE, "a") as f:
            f.write(f"CRASH at {datetime.now()}: {str(e)}\n")
    
    # Run the Cleaner (Only if raw data exists)
    if os.path.exists(CSV_FILE):
        print("\n--- Starting Machine Learning Cleaning Phase ---")
        df = pd.read_csv(CSV_FILE)
        
        # Remove duplicates (important if resuming caused overlapping saves)
        initial_len = len(df)
        df = df.drop_duplicates(subset=['appid'], keep='last')
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} duplicate records.")

        # Only keep games that have more than five reviews
        df = df[df['total_reviews'] >= 5]
        
        df = get_ratio_columns(df)
        df = get_historical_developer_prestige_metrics(df)
        df = process_release_date(df)
        
        genre_df = get_top_n_binary_df(df, 'genres', n=50, exclude_list=['Free to Play'])
        cat_df = get_top_n_binary_df(df, 'categories', n=50, exclude_list=['Free to Play'])
        
        # Merge
        df_final = pd.concat([
            df.drop(['genres', 'categories'], axis=1), 
            genre_df, 
            cat_df
        ], axis=1)

        df_final = finalize_dataset(df_final)
        df_final.to_csv(FINAL_CSV, index=False)
        print(f"Success! ML-ready dataset saved to: {FINAL_CSV} ({len(df_final)} games)")