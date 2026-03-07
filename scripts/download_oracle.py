"""
Download all historical data from Oracle's Elixir.
They have CSVs with complete match and player data since 2014.
"""

import httpx
import gzip
import shutil
from pathlib import Path
from datetime import datetime

BRONZE_PATH = Path("data/bronze/oracle_elixir")
BRONZE_PATH.mkdir(parents=True, exist_ok=True)

# Oracle's Elixir data URLs
# They provide yearly CSV files with all match data
BASE_URL = "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com"

# Available datasets
DATASETS = {
    # Match-level data
    "2024_match_data": f"{BASE_URL}/2024_LoL_esports_match_data_from_OraclesElixir.csv",
    "2023_match_data": f"{BASE_URL}/2023_LoL_esports_match_data_from_OraclesElixir.csv",
    "2022_match_data": f"{BASE_URL}/2022_LoL_esports_match_data_from_OraclesElixir.csv",
    "2021_match_data": f"{BASE_URL}/2021_LoL_esports_match_data_from_OraclesElixir.csv",
    "2020_match_data": f"{BASE_URL}/2020_LoL_esports_match_data_from_OraclesElixir.csv",
    "2019_match_data": f"{BASE_URL}/2019_LoL_esports_match_data_from_OraclesElixir.csv",
    "2018_match_data": f"{BASE_URL}/2018_LoL_esports_match_data_from_OraclesElixir.csv",
    "2017_match_data": f"{BASE_URL}/2017_LoL_esports_match_data_from_OraclesElixir.csv",
    "2016_match_data": f"{BASE_URL}/2016_LoL_esports_match_data_from_OraclesElixir.csv",
    "2015_match_data": f"{BASE_URL}/2015_LoL_esports_match_data_from_OraclesElixir.csv",
    "2014_match_data": f"{BASE_URL}/2014_LoL_esports_match_data_from_OraclesElixir.csv",
}

def download_file(name: str, url: str) -> bool:
    """Download a file from URL."""
    output_path = BRONZE_PATH / f"{name}.csv"
    
    if output_path.exists():
        print(f"  {name}: Already exists, skipping")
        return True
    
    print(f"  {name}: Downloading...")
    
    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(url)
            
            if response.status_code == 200:
                # Check if it's gzipped
                content = response.content
                
                # Try to detect if content is gzipped
                if content[:2] == b'\x1f\x8b':
                    import gzip
                    content = gzip.decompress(content)
                
                with open(output_path, "wb") as f:
                    f.write(content)
                
                size_mb = len(content) / (1024 * 1024)
                print(f"  {name}: Downloaded ({size_mb:.1f} MB)")
                return True
            else:
                print(f"  {name}: HTTP {response.status_code}")
                return False
                
    except Exception as e:
        print(f"  {name}: Error - {e}")
        return False


def main():
    print("=" * 70)
    print("ORACLE'S ELIXIR DATA DOWNLOAD")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\nOutput directory: {BRONZE_PATH}")
    print(f"Datasets to download: {len(DATASETS)}")
    print()
    
    success = 0
    failed = 0
    
    for name, url in DATASETS.items():
        if download_file(name, url):
            success += 1
        else:
            failed += 1
    
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    
    # Show what we got
    print("\nFiles downloaded:")
    total_size = 0
    for f in sorted(BRONZE_PATH.glob("*.csv")):
        size = f.stat().st_size / (1024 * 1024)
        total_size += size
        print(f"  {f.name}: {size:.1f} MB")
    
    print(f"\nTotal: {total_size:.1f} MB")


if __name__ == "__main__":
    main()
