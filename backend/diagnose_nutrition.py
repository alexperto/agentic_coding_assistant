"""
Diagnostic script to identify why NutritionTool returns
"I am currently unable to retrieve the information from the nutrition expert."

Run this with: uv run python backend/diagnose_nutrition.py
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from token_manager import create_token_manager_from_env
from search_tools import NutritionTool

def main():
    print("=" * 70)
    print("NUTRITION TOOL DIAGNOSTICS")
    print("=" * 70)

    # Check 1: Environment variables
    print("\n[1/5] Checking environment variables...")
    client_id = os.getenv('VERSA_CLIENT_ID')
    client_secret = os.getenv('VERSA_CLIENT_SECRET')
    token_url = os.getenv('OKTA_TOKEN_URL', 'https://uc-sf.okta.com/oauth2/ausnwf6tyaq6v47QF5d7/v1/token')

    print(f"   VERSA_CLIENT_ID: {'✅ SET' if client_id else '❌ NOT SET'}")
    print(f"   VERSA_CLIENT_SECRET: {'✅ SET' if client_secret else '❌ NOT SET'}")
    print(f"   OKTA_TOKEN_URL: {token_url}")

    if not client_id or not client_secret:
        print("\n❌ ERROR: OAuth credentials not configured in .env file!")
        print("   Add VERSA_CLIENT_ID and VERSA_CLIENT_SECRET to your .env file")
        return

    # Check 2: Token manager creation
    print("\n[2/5] Creating token manager...")
    token_manager = create_token_manager_from_env()

    if token_manager is None:
        print("   ❌ ERROR: create_token_manager_from_env() returned None")
        print("   This means credentials are not properly configured")
        return
    else:
        print("   ✅ Token manager created successfully")

    # Check 3: Token retrieval
    print("\n[3/5] Attempting to retrieve OAuth token...")
    try:
        token = token_manager.get_token()
        print(f"   ✅ Token retrieved successfully")
        print(f"   Token preview: {token[:30]}... (truncated for security)")
    except Exception as e:
        print(f"   ❌ ERROR retrieving token: {e}")
        print(f"   This suggests OAuth authentication is failing")
        print(f"   Check your VERSA_CLIENT_ID and VERSA_CLIENT_SECRET")
        return

    # Check 4: NutritionTool creation
    print("\n[4/5] Creating NutritionTool...")
    tool = NutritionTool(token_manager)
    print(f"   ✅ NutritionTool created")
    print(f"   API Endpoint: {tool.api_endpoint}")

    # Check 5: Execute a test question
    print("\n[5/5] Executing test nutrition question...")
    print("   Question: 'What vitamins are in spinach?'")
    print("   Calling UCSF API...")

    result = tool.execute(question="What vitamins are in spinach?")

    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(result)
    print("=" * 70)

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)

    if "Error" in result:
        print("❌ ISSUE DETECTED: Response contains an error message")
        print("\nPossible causes:")
        print("  1. API authentication failure (401)")
        print("  2. API timeout or network error")
        print("  3. API returned unexpected response format")
        print("  4. Token is invalid or expired")

        if "Authentication not configured" in result:
            print("\n⚠️ SPECIFIC ISSUE: Token manager was None")
            print("   Check token manager initialization in RAG system")
        elif "401" in result or "Unauthorized" in result:
            print("\n⚠️ SPECIFIC ISSUE: OAuth token rejected by API")
            print("   Verify your VERSA_CLIENT_ID and VERSA_CLIENT_SECRET are correct")
        elif "timed out" in result.lower():
            print("\n⚠️ SPECIFIC ISSUE: API request timed out")
            print("   Check network connectivity to UCSF API")
        elif "Invalid response format" in result:
            print("\n⚠️ SPECIFIC ISSUE: API returned unexpected JSON format")
            print("   Need to check actual API response structure")
    else:
        print("✅ SUCCESS: NutritionTool is working correctly!")
        print("   The tool returned a valid nutrition response")

    print("\n" + "=" * 70)
    print("For more details, check the test results in:")
    print("  backend/tests/NUTRITION_DIAGNOSTIC_REPORT.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
