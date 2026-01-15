#!/usr/bin/env python3
"""
Script to download public events CSV data from OpenDataSoft.
Downloads French public events data updated in 2026.
"""

import requests
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URL of the CSV file
CSV_URL = (
    #"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/csv/?limit=10&delimiters=%3B&lang=fr&refine=location_city%3AParis&refine=location_countrycode%3AFR&timezone=Europe%2FParis&use_labels=true&refine=firstdate_begin%3A2026"
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/csv/?delimiters=%3B&lang=fr&refine=location_city%3AParis&refine=location_countrycode%3AFR&timezone=Europe%2FParis&use_labels=true&refine=firstdate_begin%3A2026"
    )

# Output directory and filename
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILENAME = "evenements_publics_openagenda.csv"


def download_csv(url: str, output_path: Path) -> bool:
    """
    Download CSV file from URL and save to specified path.

    Args:
        url: URL of the CSV file to download
        output_path: Path where the file should be saved

    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading CSV from: {url}")

        # Make the request with streaming for large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size = output_path.stat().st_size
        logger.info(f"Successfully downloaded {file_size:,} bytes to {output_path}")

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download CSV: {e}")
        return False
    except IOError as e:
        logger.error(f"Failed to write file: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def main():
    """Main function to download the CSV file."""
    logger.info("Starting CSV download script")

    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    logger.info(f"Output path: {output_path}")

    success = download_csv(CSV_URL, output_path)

    if success:
        logger.info("Download completed successfully!")
        sys.exit(0)
    else:
        logger.error("Download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
