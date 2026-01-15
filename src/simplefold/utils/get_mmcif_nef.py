#!/usr/bin/env python3
# Script to download mmCIF files from RCSB PDB

import gzip
import time
import logging
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

# --- Configuration ---
BASE_MMCIF_URL = "https://files.rcsb.org/download/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

THREADS = 10
RETRY_LIMIT = 3
TIMEOUT = 30

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def read_pdb_ids(file_path: str) -> list[str]:
    """Read PDB IDs from a text file (one per line)."""
    pdb_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            pdb_id = line.strip()
            if pdb_id:  # Skip empty lines
                pdb_ids.append(pdb_id.lower())
    
    logging.info(f"Read {len(pdb_ids)} PDB IDs from {file_path}")
    return pdb_ids

def download_mmcif(pdb_id: str, output_dir: Path) -> bool:
    """Download mmCIF file for a given PDB ID."""
    pdb_id = pdb_id.lower()
    
    # mmCIF files are available in two formats:
    # 1. Compressed: {pdb_id}.cif.gz
    # 2. Uncompressed: {pdb_id}.cif
    # We'll download the compressed version and decompress it
    
    cif_gz = f"{pdb_id}.cif.gz"
    url = urljoin(BASE_MMCIF_URL, cif_gz)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdb_id}.cif"
    
    # Skip if already exists
    if out_path.exists():
        return False
    
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=TIMEOUT) as resp:
                with gzip.GzipFile(fileobj=resp) as gz:
                    out_path.write_bytes(gz.read())
            return True
            
        except Exception as e:
            if attempt == RETRY_LIMIT:
                logging.warning(f"Failed to download mmCIF for {pdb_id} after {RETRY_LIMIT} attempts: {e}")
                return False
            time.sleep(1.0 * attempt)
    
    return False

def main(pdb_id_file: str, output_dir: str):
    """Main function to download mmCIF files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read PDB IDs from file
    pdb_ids = read_pdb_ids(pdb_id_file)
    
    if not pdb_ids:
        logging.error("No PDB IDs found in the input file!")
        return
    
    # Download files in parallel
    total_downloaded = 0
    successful_ids = []
    
    logging.info(f"Starting parallel downloads with {THREADS} threads...")
    
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(download_mmcif, pid, output_path): pid for pid in pdb_ids}
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Downloading mmCIF files"):
            pid = futures[future]
            try:
                if future.result():
                    total_downloaded += 1
                    successful_ids.append(pid)
            except Exception as e:
                logging.error(f"Error processing {pid}: {e}")
    
    # Write successful downloads list
    success_list_path = output_path / "downloaded_pdb_ids.txt"
    with open(success_list_path, "w") as f:
        f.write("\n".join(sorted(successful_ids)))
    
    logging.info("✓ Download Complete")
    logging.info(f"✓ Total mmCIF files downloaded: {total_downloaded}")
    logging.info(f"✓ Already existing files: {len(pdb_ids) - total_downloaded}")
    logging.info(f"✓ Downloaded IDs saved: {success_list_path.resolve()}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <pdb_id_file.txt> <output_dir>")
        print(f"Example: python {sys.argv[0]} pdb_ids.txt ./mmcif_files")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])