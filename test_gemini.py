"""
Diagnostic script to test Gemini API models using google-genai and GEMINI_API_KEY.
"""

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "").strip()

print("=" * 60)
print("Gemini API Diagnostic Test")
print("=" * 60)
print(f"API Key present: {bool(api_key)}")
if api_key:
    print(f"API Key prefix: {api_key[:8]}... (Length: {len(api_key)})")
else:
    print("ERROR: GEMINI_API_KEY is missing from .env!")

models_to_test = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp"
]

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    print("\nAttempting calls with google-genai client:\n")

    working_model = None
    for model_name in models_to_test:
        print(f"Testing model: '{model_name}'...")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Hello, reply with 'OK' only."
            )
            text = (response.text or "").strip()
            print(f"  --> SUCCESS! Response: '{text}'")
            working_model = model_name
            break
        except Exception as e:
            print(f"  --> FAILED: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    if working_model:
        print(f"RESULT: Working model found: '{working_model}'")
    else:
        print("RESULT: All tested Gemini models failed.")
        print("Action Required: Please create a new API key at https://aistudio.google.com")
        print("using 'Create API key in new project'.")
    print("=" * 60)

except ImportError as e:
    print(f"ImportError: {e}. Please ensure google-genai is installed.")
