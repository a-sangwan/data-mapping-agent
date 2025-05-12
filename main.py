import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from groq import Groq
from langgraph.graph import END, StateGraph

# Configure GROQ
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_data():
    with open("sample_data.json", "r") as f:
        return json.load(f)

def extract_from_name(product_name):
    """Extract product attributes from name using LLM"""
    prompt = f"""Extract product information from this name: "{product_name}"

Return JSON with:
- brand: extracted brand name (pepsi, lays, doritos, etc)
- category: product category (carbonated drinks, snacks, sports drinks, juices, cereals)
- size: size/volume (330ml, 40g, 500ml, etc)
- flavor: flavor or variant (cherry, bbq, nacho cheese, etc)
- product_type: main product type (cola, chips, juice, etc)

Example: "Cherry Pepsi 330ml Can" -> {{"brand": "pepsi", "category": "carbonated drinks", "size": "330ml", "flavor": "cherry", "product_type": "cola"}}

Respond only with valid JSON."""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        # Fallback to simple keyword matching
        name_lower = product_name.lower()
        return {
            "brand": next((b for b in ["pepsi", "lays", "doritos", "gatorade", "tropicana", "quaker"] if b in name_lower), ""),
            "category": next((c for c in ["carbonated drinks", "snacks", "sports drinks", "juices", "cereals"] if any(k in name_lower for k in c.split())), "unknown"),
            "size": "",
            "flavor": "",
            "product_type": ""
        }

def find_matches(extracted_info, product_name, internal_catalog):
    """Find matching internal products"""
    candidates = []

    for internal_product in internal_catalog:
        score = 0.0
        internal_name = internal_product.get("name", "").lower()

        # Brand match
        if extracted_info.get("brand") and extracted_info["brand"].lower() in internal_name:
            score += 0.3

        # Category match
        if extracted_info.get("category") and extracted_info["category"] == internal_product.get("category", ""):
            score += 0.3

        # Size match
        if extracted_info.get("size") and extracted_info["size"] in internal_name:
            score += 0.2

        # Flavor match
        if extracted_info.get("flavor") and extracted_info["flavor"].lower() in internal_name:
            score += 0.2

        # Product type match
        if extracted_info.get("product_type") and extracted_info["product_type"].lower() in internal_name:
            score += 0.2

        # Direct name keyword matching
        original_words = product_name.lower().split()
        for word in original_words:
            if len(word) > 2 and word in internal_name:  # Skip short words
                score += 0.1

        if score > 0.3:  # Threshold for candidate
            candidates.append({
                "internal_product": internal_product,
                "score": score,
                "match_reason": f"Brand: {extracted_info.get('brand', 'N/A')}, Category: {extracted_info.get('category', 'N/A')}"
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:5]

def main():
    data = load_data()
    mappings = []

    # Process external products with only names
    for external_product in data["external_products"]:
        product_name = external_product.get("name", "")
        print(f"\nProcessing: {product_name}")

        # Extract attributes from name
        extracted_info = extract_from_name(product_name)
        print(f"Extracted: {extracted_info}")

        # Find matches
        candidates = find_matches(extracted_info, product_name, data["internal_catalog"])

        if candidates:
            best_match = candidates[0]
            print(f"Best match: {best_match['internal_product']['name']}")
            print(f"Score: {best_match['score']:.2f}")
            print(f"Reason: {best_match['match_reason']}")

            # Add to mappings
            mappings.append({
                "external_id": external_product.get("id"),
                "external_name": product_name,
                "internal_id": best_match['internal_product']['id'],
                "internal_name": best_match['internal_product']['name'],
                "confidence_score": best_match['score'],
                "match_reason": best_match['match_reason']
            })
        else:
            print("No match found")

    return mappings

if __name__ == "__main__":
    output_mapping = main()
    print(output_mapping)
