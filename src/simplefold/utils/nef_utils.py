#!/usr/bin/env python3
# Copyright 2025 by Imesh Ranaweera, Alberto Perez
# All rights reserved

import json
import gzip
import time
import logging
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
import requests

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
BASE_DIVIDED_URL = "https://files.rcsb.org/pub/pdb/data/structures/divided/nmr_data/"
HEADERS = {"User-Agent": "Mozilla/5.0"}


THREADS = 48         
RETRY_LIMIT = 3       
TIMEOUT = 20       



logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)


def fetch_nmr_pdb_ids():
    """Fetch all PDB IDs solved by solution NMR using RCSB POST API."""

    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": "solution NMR"
            }
        },
        "request_options": {"return_all_hits": True},
        "return_type": "entry"
    }

    logging.info("Querying RCSB API for NMR PDB IDs...")

    resp = requests.post(
        RCSB_SEARCH_URL,
        json=query,
        headers=HEADERS,
        timeout=TIMEOUT
    )

    resp.raise_for_status()
    data = resp.json()

    results = [r["identifier"] for r in data.get("result_set", [])]

    logging.info(f"Found {len(results)} NMR entries.")
    return results



def download_one(pdb_id: str, output_dir: Path):
    """Download and decompress one NEF file (with retries)."""

    pdb_id = pdb_id.lower()
    subfolder = pdb_id[1:3]

    nef_gz = f"{pdb_id}_nmr-data.nef.gz"
    url_folder = urljoin(BASE_DIVIDED_URL, f"{subfolder}/")
    url_file = urljoin(url_folder, nef_gz)

    out_path = output_dir / f"{pdb_id}.nef"

    if out_path.exists():
        return False  # Already downloaded

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            req = Request(url_file, headers=HEADERS)
            with urlopen(req, timeout=TIMEOUT) as resp:
                with gzip.GzipFile(fileobj=resp) as gz:
                    out_path.write_bytes(gz.read())
            return True

        except Exception:
            if attempt == RETRY_LIMIT:
                return False
            time.sleep(1.0 * attempt)  # exponential backoff


def main(output_path: str):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = fetch_nmr_pdb_ids()
    if not pdb_ids:
        logging.error("No PDB IDs fetched — aborting.")
        return

    total = 0

    logging.info(f"Starting parallel downloads with {THREADS} threads...")

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(download_one, pid, output_dir): pid for pid in pdb_ids}

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            try:
                if future.result():
                    total += 1
            except Exception:
                pass

    logging.info(f" NEF Download Complete")
    logging.info(f" Total downloaded: {total}")
    logging.info(f" Saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <output_dir>")
        sys.exit(1)
    main(sys.argv[1])
