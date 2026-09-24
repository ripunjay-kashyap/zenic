"""
USDA FoodData Central ingestion.

Two modes:
  1. Bulk file  — parse a downloaded Foundation Foods or SR Legacy JSON
                  (preferred: https://fdc.nal.usda.gov/download-foods.html)
  2. API list   — paginate through /foods/list to grab the top N foods
                  (slower but requires no manual download)

Chunking: one document per food item with its full nutrient profile.
Metadata: category, food_group, source="USDA", fdcId, data_type
"""
import json
import os
import time
from pathlib import Path

import httpx

from zenic.logging_config import get_logger

logger = get_logger(__name__)

_BASE = "https://api.nal.usda.gov/fdc/v1"

# Nutrients we care about — keeps chunk size reasonable
_NUTRIENT_NAMES = {
    "Energy",
    "Protein",
    "Total lipid (fat)",
    "Carbohydrate, by difference",
    "Fiber, total dietary",
    "Sugars, total including NLEA",
    "Calcium, Ca",
    "Iron, Fe",
    "Magnesium, Mg",
    "Phosphorus, P",
    "Potassium, K",
    "Sodium, Na",
    "Zinc, Zn",
    "Copper, Cu",
    "Selenium, Se",
    "Vitamin C, total ascorbic acid",
    "Thiamin",
    "Riboflavin",
    "Niacin",
    "Vitamin B-6",
    "Folate, total",
    "Vitamin B-12",
    "Vitamin A, RAE",
    "Vitamin E (alpha-tocopherol)",
    "Vitamin D (D2 + D3)",
    "Vitamin K (phylloquinone)",
    "Fatty acids, total saturated",
    "Fatty acids, total monounsaturated",
    "Fatty acids, total polyunsaturated",
    "Cholesterol",
}


def _format_food_doc(food: dict) -> dict | None:
    """Convert a USDA food dict into an indexable document."""
    description = food.get("description", "").strip()
    if not description:
        return None

    fdc_id = food.get("fdcId") or food.get("id")
    category = food.get("foodCategory", {})
    if isinstance(category, dict):
        category_name = category.get("description", "")
    else:
        category_name = str(category)

    food_group = food.get("foodGroup", category_name)
    data_type = food.get("dataType", "")

    # Build nutrient table text
    nutrients = food.get("foodNutrients", [])
    nutrient_lines = []
    for n in nutrients:
        # Handle all USDA API response formats:
        #  - bulk JSON: {"nutrient": {"name": ...}, "amount": ...}
        #  - /foods/list: {"name": ..., "amount": ...}
        #  - /foods/search: {"nutrientName": ..., "value": ...}
        nutrient_info = n.get("nutrient", {})
        name = (
            nutrient_info.get("name")
            or n.get("name")
            or n.get("nutrientName", "")
        )
        if name not in _NUTRIENT_NAMES:
            continue
        value = n.get("amount", n.get("value"))
        unit = nutrient_info.get("unitName", n.get("unitName", ""))
        if value is not None:
            nutrient_lines.append(f"  {name}: {value} {unit}")

    if not nutrient_lines:
        return None

    parts = [
        f"Food: {description}",
        f"Category: {category_name}" if category_name else None,
        f"Data type: {data_type}" if data_type else None,
        "Nutrients per 100g:",
        *nutrient_lines,
    ]
    text = "\n".join(p for p in parts if p is not None)

    return {
        "id": f"usda_{fdc_id}",
        "text": text,
        "metadata": {
            "source": "USDA",
            "fdcId": fdc_id,
            "food_name": description,
            "category": category_name,
            "food_group": food_group,
            "data_type": data_type,
        },
    }


def ingest_usda_bulk(json_path: str) -> list[dict]:
    """
    Parse a downloaded USDA FoodData Central bulk JSON file.
    Supports both the Foundation Foods and SR Legacy JSON formats.

    Download from: https://fdc.nal.usda.gov/download-foods.html
    Recommended: FoundationDownload.zip or FoodData_Central_sr_legacy_food_json_YYYY-MM-DD.zip
    """
    path = Path(json_path)
    logger.info(f"Loading USDA bulk file: {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # The JSON root is typically {"FoundationFoods": [...]} or {"SRLegacyFoods": [...]}
    foods = []
    for key in ("FoundationFoods", "SRLegacyFoods", "foods"):
        if key in data:
            foods = data[key]
            break
    if not foods and isinstance(data, list):
        foods = data

    logger.info(f"  {len(foods)} foods in bulk file")
    docs = [_format_food_doc(f) for f in foods]
    docs = [d for d in docs if d is not None]
    logger.info(f"  {len(docs)} documents prepared")
    return docs


def ingest_usda_api(n: int = 3000, data_types: list[str] | None = None) -> list[dict]:
    """
    Paginate through the USDA FoodData Central /foods/list API.
    Use when no bulk file is available. Slower (~5-10 minutes for 3000 foods).

    data_types: list of USDA data types to filter by, e.g.
                ["Foundation", "SR Legacy", "Survey (FNDDS)"]
                Defaults to Foundation + SR Legacy (highest quality nutritional data).
    """
    api_key = os.environ["USDA_API_KEY"]
    if data_types is None:
        data_types = ["Foundation", "SR Legacy"]

    docs = []
    page = 1
    page_size = 200

    logger.info(f"Fetching top {n} USDA foods via API (data types: {data_types})...")

    with httpx.Client(timeout=30) as client:
        while len(docs) < n:
            resp = client.post(
                f"{_BASE}/foods/list",
                params={"api_key": api_key},
                json={
                    "dataType": data_types,
                    "pageSize": page_size,
                    "pageNumber": page,
                },
            )
            if resp.status_code == 429:
                logger.info("  Rate limited — waiting 5s...")
                time.sleep(5)
                continue
            resp.raise_for_status()
            foods = resp.json()
            if not foods:
                break

            batch_docs = [_format_food_doc(f) for f in foods]
            batch_docs = [d for d in batch_docs if d is not None]
            docs.extend(batch_docs)
            logger.info(f"  Page {page}: fetched {len(batch_docs)} docs (total: {len(docs)})")
            page += 1

            if len(foods) < page_size:
                break  # last page

    docs = docs[:n]
    logger.info(f"Prepared {len(docs)} USDA food documents")
    return docs
