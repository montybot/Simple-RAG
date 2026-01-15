# src/csv_processor.py
import pandas as pd
from pathlib import Path
from typing import List, Dict
from loguru import logger


class CSVProcessor:
    """Enhanced CSV processor that preserves structured fields."""

    def __init__(self):
        """Initialize the CSV processor."""
        # Key fields to extract and make searchable
        self.key_fields = {
            'Identifiant': 'ID',
            'Titre': 'Title',
            'Description': 'Description',
            'Description longue': 'Long Description',
            'Mots clés': 'Keywords',
            'Première date - Début': 'Start Date',
            'Première date - Fin': 'End Date',
            'Résumé horaires': 'Schedule',
            'Nom du lieu': 'Venue',
            'Adresse': 'Address',
            'Code postal': 'Postal Code',
            'Ville': 'City',
            'Département': 'Department',
            'Région': 'Region',
            'Catégorie': 'Category',
            'Accessibilité': 'Accessibility',
            'Détail des conditions': 'Conditions',
        }

    def process_csv(self, file_path: Path) -> List[Dict]:
        """
        Process a CSV file into structured chunks.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of chunk dictionaries with text and metadata
        """
        logger.info(f"Processing CSV file: {file_path}")

        try:
            # Read CSV with pandas (handle semicolon delimiter)
            df = pd.read_csv(file_path, sep=';', encoding='utf-8-sig')
            logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")

            chunks = []

            # Process each row as a separate event
            for idx, row in df.iterrows():
                chunk_text = self._format_event(row)
                metadata = self._extract_metadata(row, file_path)

                chunks.append({
                    'text': chunk_text,
                    'metadata': metadata
                })

            logger.info(f"Created {len(chunks)} structured chunks from CSV")
            return chunks

        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            raise

    def _format_event(self, row: pd.Series) -> str:
        """
        Format a CSV row into a structured text chunk.

        Args:
            row: Pandas Series representing one row

        Returns:
            Formatted text string
        """
        parts = []

        # Build structured text with clear labels
        for csv_field, english_label in self.key_fields.items():
            if csv_field in row and pd.notna(row[csv_field]) and str(row[csv_field]).strip():
                value = str(row[csv_field]).strip()
                # Add the field with its label for better semantic understanding
                parts.append(f"**{english_label}**: {value}")

        # Join all parts with newlines
        return "\n".join(parts)

    def _extract_metadata(self, row: pd.Series, file_path: Path) -> Dict:
        """
        Extract metadata from a CSV row.

        Args:
            row: Pandas Series representing one row
            file_path: Original file path

        Returns:
            Metadata dictionary
        """
        metadata = {
            'source_file': str(file_path),
            'source_type': 'csv',
            'event_id': str(row.get('Identifiant', '')),
        }

        # Add key fields as metadata for potential filtering
        if 'Ville' in row and pd.notna(row['Ville']):
            metadata['city'] = str(row['Ville'])

        if 'Région' in row and pd.notna(row['Région']):
            metadata['region'] = str(row['Région'])

        if 'Première date - Début' in row and pd.notna(row['Première date - Début']):
            metadata['start_date'] = str(row['Première date - Début'])

        if 'Catégorie' in row and pd.notna(row['Catégorie']):
            metadata['category'] = str(row['Catégorie'])

        if 'Titre' in row and pd.notna(row['Titre']):
            metadata['title'] = str(row['Titre'])

        return metadata
