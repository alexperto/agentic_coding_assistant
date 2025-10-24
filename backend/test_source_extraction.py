"""
Test script to verify NutritionTool source extraction
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from search_tools import NutritionTool

# Sample response from the API (the actual format we saw working)
sample_answer = """Spinach contains several vitamins, including:

- **Vitamin A**: Essential for vision, immune function, and skin health.
- **Vitamin C**: Important for immune support, collagen production, and antioxidant protection.
- **Vitamin K**: Crucial for blood clotting and bone health.
- **Folate (Vitamin B9)**: Supports cell growth and DNA synthesis.

Cited Sources:
<a href="/get_document/versaplatformprod/versaplatform/eureka_fim/eureka_fim-doc/Vegetables > Spinach and Corn Pancakes.pdf" target="_blank">Vegetables > Spinach and Corn Pancakes.pdf</a>"""

print("="*70)
print("TESTING NUTRITION TOOL SOURCE EXTRACTION")
print("="*70)

# Create NutritionTool instance (with None token manager for this test)
tool = NutritionTool(None)

# Test the _extract_sources method
cleaned_answer, sources = tool._extract_sources(sample_answer)

print("\n[1] Original Answer:")
print("-" * 70)
print(sample_answer)

print("\n[2] Cleaned Answer (without sources section):")
print("-" * 70)
print(cleaned_answer)

print("\n[3] Extracted Sources:")
print("-" * 70)
for i, source in enumerate(sources, 1):
    print(f"\nSource {i}:")
    print(f"  Text: {source['text']}")
    print(f"  URL: {source['url']}")

print("\n" + "="*70)
print("VERIFICATION:")
print("="*70)

# Verify cleaned answer doesn't contain "Cited Sources"
has_cited_sources = "Cited Sources" in cleaned_answer or "cited sources" in cleaned_answer.lower()
print(f"✅ Cleaned answer removed 'Cited Sources' section: {not has_cited_sources}")

# Verify sources were extracted
print(f"✅ Number of sources extracted: {len(sources)}")

# Verify source format
if sources:
    has_text = all('text' in s for s in sources)
    has_url = all('url' in s for s in sources)
    print(f"✅ All sources have 'text' field: {has_text}")
    print(f"✅ All sources have 'url' field: {has_url}")

    # Verify URLs are absolute
    all_absolute = all(s['url'].startswith('http') for s in sources)
    print(f"✅ All URLs are absolute: {all_absolute}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
