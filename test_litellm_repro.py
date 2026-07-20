#!/usr/bin/env python3
"""
Accurate reproduction of the litellm function_call_output issue.

This script mimics exactly what litellm sends to the Responses API endpoint,
based on the transformation code in litellm/responses/transformation.py.
"""
import argparse
import httpx
import json
import sys
import time

parser = argparse.ArgumentParser(description="Agentic-api function_call_output reproduction test")
parser.add_argument("--url", required=True, help="API base URL (e.g., http://localhost:9000/v1)")
args = parser.parse_args()
BASE_URL = args.url

def test_simple_text_input():
    """Simple text input to verify endpoint works."""
    print("=" * 60)
    print("TEST 1: Simple text input (baseline)")
    print("=" * 60)
    
    # Minimal valid input per OpenAI Responses API
    payload = {
        "model": "qwen3-27b",
        "input": [{"role": "user", "content": "Hello"}]
    }
    
    print(f"Payload: {json.dumps(payload)}")
    
    try:
        resp = httpx.post(f"{BASE_URL}/responses", json=payload, timeout=60)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print("✅ SUCCESS: Basic endpoint works")
            return True
        else:
            print(f"❌ FAIL: {resp.text[:500]}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False


def test_text_input_with_content_parts():
    """Test with content as array of parts."""
    print("\n" + "=" * 60)
    print("TEST 2: Text input with content parts")
    print("=" * 60)
    
    payload = {
        "model": "qwen3-27b",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
        ]
    }
    
    print(f"Payload: {json.dumps(payload)}")
    
    try:
        resp = httpx.post(f"{BASE_URL}/responses", json=payload, timeout=60)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print("✅ SUCCESS: Content parts work")
            return True
        else:
            print(f"❌ FAIL: {resp.text[:500]}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False


def test_function_call_output_string():
    """Test with function_call_output where output is STRING."""
    print("\n" + "=" * 60)
    print("TEST 3: function_call_output with STRING output")
    print("=" * 60)
    
    payload = {
        "model": "qwen3-27b",
        "input": [
            {"role": "user", "content": "What files are in the directory?"},
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "/workspace contains: README.md, src/"
            }
        ]
    }
    
    print(f"Payload: {json.dumps(payload)}")
    
    try:
        resp = httpx.post(f"{BASE_URL}/responses", json=payload, timeout=60)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print("✅ SUCCESS: String output accepted")
            return True
        else:
            print(f"❌ FAIL: {resp.text[:500]}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False


def test_function_call_output_list():
    """Test with function_call_output where output is LIST - the failing case."""
    print("\n" + "=" * 60)
    print("TEST 4: function_call_output with LIST output (litellm format)")
    print("=" * 60)
    
    payload = {
        "model": "qwen3-27b",
        "input": [
            {"role": "user", "content": "What files are in the directory?"},
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": [
                    {"type": "input_text", "text": "/workspace contains: README.md, src/"}
                ]
            }
        ]
    }
    
    print(f"Payload: {json.dumps(payload)}")
    
    try:
        resp = httpx.post(f"{BASE_URL}/responses", json=payload, timeout=60)
        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"\n🔴 FAILURE (reproduced): {resp.status_code}")
            print(f"Error: {resp.text[:1000]}")
            return True
        else:
            print("\n✅ SUCCESS: List output accepted!")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return None


def test_multi_turn_with_tool_call():
    """Test a multi-turn conversation with a tool call and result."""
    print("\n" + "=" * 60)
    print("TEST 5: Multi-turn with tool call and list output")
    print("=" * 60)
    
    # First turn
    turn1 = {
        "model": "qwen3-27b",
        "input": [{"role": "user", "content": "What files are in /workspace?"}]
    }
    
    print("Turn 1 payload:", json.dumps(turn1))
    resp1 = httpx.post(f"{BASE_URL}/responses", json=turn1, timeout=60)
    print(f"Status: {resp1.status_code}")
    
    if resp1.status_code != 200:
        print(f"❌ Turn 1 failed: {resp1.text[:500]}")
        return None
    
    data1 = resp1.json()
    print(f"Response ID: {data1.get('id')}")
    
    # Extract tool call if present
    tool_call = None
    for item in data1.get("output", []):
        if item.get("type") == "function_call":
            tool_call = item
            break
    
    if not tool_call:
        print("No tool call in response")
        return False
    
    print(f"Tool call: {tool_call}")
    tool_call_id = tool_call.get("id") or tool_call.get("call_id")
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("arguments")
    
    # Second turn with list output
    turn2 = {
        "model": "qwen3-27b",
        "input": [
            {"role": "user", "content": "What files are in /workspace?"},
            {
                "role": "assistant",
                "content": [{
                    "type": "function_call",
                    "call_id": tool_call_id,
                    "name": tool_name,
                    "arguments": tool_args
                }]
            },
            {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": [{"type": "input_text", "text": "/workspace contains: README.md, src/"}]
            }
        ]
    }
    
    print("\nTurn 2 payload:", json.dumps(turn2))
    resp2 = httpx.post(f"{BASE_URL}/responses", json=turn2, timeout=60)
    print(f"Status: {resp2.status_code}")
    
    if resp2.status_code != 200:
        print(f"\n🔴 FAILURE (reproduced): {resp2.status_code}")
        print(f"Error: {resp2.text[:1000]}")
        return True
    else:
        print("\n✅ SUCCESS: Multi-turn with list output works!")
        return False


if __name__ == "__main__":
    print("Agentic-api litellm function_call_output reproduction")
    print(f"Target: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run all tests
    r1 = test_simple_text_input()
    r2 = test_text_input_with_content_parts()
    r3 = test_function_call_output_string()
    r4 = test_function_call_output_list()
    r5 = test_multi_turn_with_tool_call()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Simple text: {'PASS' if r1 else 'FAIL'}")
    print(f"Content parts: {'PASS' if r2 else 'FAIL'}")
    print(f"String output: {'PASS' if r3 else 'FAIL'}")
    print(f"List output: {'REPRODUCED' if r4 else 'FIXED' if r4 is False else 'UNKNOWN'}")
    print(f"Multi-turn: {'REPRODUCED' if r5 else 'FIXED' if r5 is False else 'N/A'}")
    
    # Determine overall result
    if r4 or r5:  # If either list test failed
        print("\n🔴 Issue REPRODUCED: Server rejects list output in function_call_output")
        sys.exit(1)
    elif r4 is False or r5 is False:  # If either list test succeeded
        print("\n🟢 Issue FIXED: Server accepts list output")
        sys.exit(0)
    else:
        print("\n🟡 Test inconclusive")
        sys.exit(2)
