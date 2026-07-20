#!/usr/bin/env python3
"""
Diagnostic test to determine what input format the server accepts.
Tests multiple variations to understand the server's schema.
"""
import argparse
import httpx
import json
import sys
import time

parser = argparse.ArgumentParser(description="Server input format diagnostic test")
parser.add_argument("--url", required=True, help="API base URL (e.g., http://localhost:9000/v1)")
args = parser.parse_args()
BASE_URL = args.url

def test_format(name, payload):
    """Test a specific payload format."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Payload: {json.dumps(payload)[:200]}...")
    
    try:
        resp = httpx.post(f"{BASE_URL}/responses", json=payload, timeout=30)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print("✅ SUCCESS")
            return True
        else:
            print(f"❌ FAIL: {resp.text[:300]}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    print(f"Server Input Format Diagnostic")
    print(f"Target: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        # Test 1: Model as string
        ("Model as string", {
            "model": "qwen3-27b",
            "input": [{"role": "user", "content": "Hello"}]
        }),
        
        # Test 2: Model as object
        ("Model as object", {
            "model": {"name": "qwen3-27b"},
            "input": [{"role": "user", "content": "Hello"}]
        }),
        
        # Test 3: Input as string
        ("Input as string", {
            "model": "qwen3-27b",
            "input": "Hello"
        }),
        
        # Test 4: Input as array of strings
        ("Input as array", {
            "model": "qwen3-27b",
            "input": ["Hello"]
        }),
        
        # Test 5: Input with different role format
        ("Role as object", {
            "model": "qwen3-27b",
            "input": [{"role": {"type": "user"}, "content": "Hello"}]
        }),
        
        # Test 6: Content as input_text type
        ("Content as input_text", {
            "model": "qwen3-27b",
            "input": [{"role": "user", "content": {"type": "input_text", "text": "Hello"}}]
        }),
        
        # Test 7: Content as array with input_text
        ("Content array input_text", {
            "model": "qwen3-27b",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
        }),
        
        # Test 8: Different field name
        ("Messages field", {
            "model": "qwen3-27b",
            "messages": [{"role": "user", "content": "Hello"}]
        }),
        
        # Test 9: Empty input
        ("Empty input", {
            "model": "qwen3-27b",
            "input": []
        }),
        
        # Test 10: With response format
        ("With response format", {
            "model": "qwen3-27b",
            "input": [{"role": "user", "content": "Hello"}],
            "response_format": {"type": "text"}
        }),
    ]
    
    results = []
    for name, payload in tests:
        result = test_format(name, payload)
        results.append((name, result))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL" if result is False else "🟡 ERROR"
        print(f"{status}: {name}")
    
    # Check if any passed
    if any(r for _, r in results):
        print("\n🟢 Some formats accepted - server is reachable")
    else:
        print("\n🔴 No formats accepted - server may be misconfigured or unreachable")
