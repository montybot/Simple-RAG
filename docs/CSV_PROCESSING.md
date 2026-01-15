# CSV Processing for RAG System

## Overview

The RAG system now includes **enhanced CSV processing** that preserves structured fields, making it easy to query specific information like dates, locations, and categories.

## How It Works

### Before (Docling default)
CSV was converted to markdown table format, losing semantic meaning:
```
| Title | City | Date |
|-------|------|------|
| Event1 | Paris | 2026-01-31 |
```

### After (Structured Processing)
Each CSV row becomes a **labeled document**:
```
**ID**: 59703165
**Title**: LE BALCON - La Planète Sauvage
**Description**: La Planète Sauvage est une merveille...
**Start Date**: 2026-01-31T20:00:00+01:00
**City**: Paris
**Region**: Île-de-France
**Venue**: Cité de la Musique - Philharmonie 2
**Address**: 221 Avenue Jean Jaurès, 75019 Paris
**Keywords**: Le Balcon, La Planète Sauvage, Ciné-Concert
**Category**: Culture
```

## Key Fields Preserved

The CSV processor extracts and labels these important fields:

| CSV Column | English Label | Purpose |
|------------|---------------|---------|
| Identifiant | ID | Unique identifier |
| Titre | Title | Event name |
| Description | Description | Short description |
| Description longue | Long Description | Full details |
| Mots clés | Keywords | Searchable tags |
| Première date - Début | Start Date | Event start datetime |
| Première date - Fin | End Date | Event end datetime |
| Résumé horaires | Schedule | Schedule summary |
| Nom du lieu | Venue | Venue name |
| Adresse | Address | Street address |
| Code postal | Postal Code | Postal code |
| Ville | City | City name |
| Département | Department | French department |
| Région | Region | French region |
| Catégorie | Category | Event category |
| Accessibilité | Accessibility | Accessibility info |
| Détail des conditions | Conditions | Price/conditions |

## Metadata Storage

Each event chunk includes searchable metadata:
```json
{
  "source_file": "/app/data/raw/evenements_publics_openagenda.csv",
  "source_type": "csv",
  "event_id": "59703165",
  "city": "Paris",
  "region": "Île-de-France",
  "start_date": "2026-01-31T20:00:00+01:00",
  "title": "LE BALCON - La Planète Sauvage"
}
```

## Query Examples

### Location-based queries
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels événements ont lieu à Paris?", "top_k": 5}'
```

**Result**: Returns events with **City: Paris**, showing venue, address, and dates.

### Date-based queries
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels événements commencent en janvier 2026?", "top_k": 5}'
```

**Result**: Returns events with **Start Date** in January 2026, properly formatted.

### Category-based queries
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels événements culturels sont disponibles?", "top_k": 5}'
```

**Result**: Returns events matching cultural **Keywords** or **Category**.

### Venue-based queries
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Où se trouve le Théâtre du Châtelet?", "top_k": 3}'
```

**Result**: Returns events with **Venue** and **Address** information.

### Combined queries
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Événements de théâtre à Paris en janvier", "top_k": 5}'
```

**Result**: Combines **City**, **Start Date**, and **Keywords** filtering.

## Files Modified

### New Files
- `src/csv_processor.py` - Enhanced CSV processor with structured field extraction

### Updated Files
- `src/document_processor.py` - Integrated CSV processor for `.csv` files

## Usage

### Rebuild Index with Structured CSV
```bash
# From inside container
python scripts/build_index.py --input-dir /app/data/raw --force-rebuild

# Or from host
docker compose -f docker/docker-compose.yml exec rag-system \
  python scripts/build_index.py --input-dir /app/data/raw --force-rebuild
```

### Query the Index
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here", "top_k": 5}'
```

## Benefits

1. **Field-aware queries**: LLM understands field labels (City, Date, Venue, etc.)
2. **Better context**: Structured format provides clearer information
3. **Accurate filtering**: Can extract specific dates, locations, categories
4. **Metadata storage**: Future support for hybrid search (semantic + filters)
5. **Scalability**: Each row is independent, easy to update/delete

## Customization

To add or modify fields, edit `src/csv_processor.py`:

```python
self.key_fields = {
    'Your_CSV_Column': 'Your Label',
    'Another_Column': 'Another Label',
    # ... add more fields
}
```

Then rebuild the index.

## Performance

- **Index building**: ~0.5-1s for 15 events
- **Query time**: ~1-3s depending on complexity
- **Chunk size**: One chunk per CSV row (event)
- **Embedding model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

## Next Steps

### Potential Enhancements

1. **Hybrid Search**: Add metadata filtering before semantic search
   ```python
   # Filter by city first, then semantic search
   results = vector_store.search(query, filters={"city": "Paris"})
   ```

2. **Date Range Queries**: Parse dates and filter by range
   ```python
   # Events between two dates
   results = filter_by_date_range(start="2026-01-01", end="2026-01-31")
   ```

3. **Geographic Search**: Use coordinates for proximity search
   ```python
   # Events near a location
   results = search_near(lat=48.8566, lon=2.3522, radius_km=5)
   ```

4. **Category Facets**: Group results by category
   ```python
   # Count events by category
   facets = get_facets(field="category")
   ```

## Troubleshooting

### CSV not loading
- Check delimiter (should be `;` for this dataset)
- Ensure UTF-8 encoding with BOM handling (`encoding='utf-8-sig'`)

### Fields not appearing
- Verify column names match exactly (case-sensitive)
- Check for empty/NaN values in CSV

### Poor query results
- Increase `top_k` parameter for more results
- Check if field labels are clear in the chunks
- Consider adjusting embedding model

## References

- CSV Processor: `src/csv_processor.py`
- Document Processor: `src/document_processor.py`
- Example CSV: `data/raw/evenements_publics_openagenda.csv`
