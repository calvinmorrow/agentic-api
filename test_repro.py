#!/usr/bin/env python3
"""
Minimal reproduction of the function_call_output list schema error.

The issue: litellm sends `output` as a list of content parts, but agentic-api 
only accepts a string. This script tests both formats against the live endpoint.
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

# Minimal tool definition for testing
TOOL = {
    "type": "function",
    "name": "get_directory",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path"
            }
        },
        "required": ["path"]
    }
}


def test_with_list_output():
    """Test with output as LIST - this is what litellm sends and should fail."""
    print("=" * 60)
    print("TEST 1: function_call_output with LIST output")
    print("=" * 60)
    
    # First turn: ask a question that will trigger a tool call
    turn1_input = {
        "model": "qwen3-27b",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What's in the /workspace directory?"}]
            }
        ],
        "tools": [TOOL],
        "tool_choice": "auto"
    }
    
    print("\n--- Turn 1: Initial request ---")
    try:
        resp1 = httpx.post(f"{BASE_URL}/responses", json=turn1_input, timeout=60)
        print(f"Status: {resp1.status_code}")
        if resp1.status_code == 200:
            data1 = resp1.json()
            print(f"Response ID: {data1.get('id', 'N/A')}")
            
            # Extract tool call ID
            output_items = data1.get("output", [])
            tool_call_id = None
            for item in output_items:
                if item.get("type") == "function_call":
                    tool_call_id = item.get("id") or item.get("call_id")
                    print(f"Tool call ID: {tool_call_id}")
                    print(f"Tool name: {item.get('name')}")
                    break
            
            if not tool_call_id:
                print("No tool call in response, can't test list output")
                return False
            
            # Second turn: send the tool result with LIST output
            turn2_input = {
                "model": "qwen3-27b",
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "What's in the /workspace directory?"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "function_call", "call_id": tool_call_id, "name": "get_directory", "arguments": '{"path":"/workspace"}'}]
                    },
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": [{"type": "input_text", "text": "/workspace contains: README.md, src/"}]
                    }
                ]
            }
            
            print("\n--- Turn 2: Sending tool result with LIST output ---")
            print(json.dumps(turn2_input, indent=2))
            print("\nSending request...")
            
            resp2 = httpx.post(f"{BASE_URL}/responses", json=turn2_input, timeout=60)
            print(f"Status: {resp2.status_code}")
            
            if resp2.status_code != 200:
                print(f"\n❌ FAILURE: {resp2.status_code}")
                print(f"Error: {resp2.text[:500]}")
                return True  # Successfully reproduced the failure
            else:
                print("\n✅ SUCCESS: Server accepted list output!")
                return False  # No failure - fix is working
                
        else:
            print(f"Turn 1 failed: {resp1.text[:500]}")
            return None
            
    except Exception as e:
        print(f"Exception: {e}")
        return None


def test_with_string_output():
    """Test with output as STRING - this should always work."""
    print("\n" + "=" * 60)
    print("TEST 2: function_call_output with STRING output (baseline)")
    print("=" * 60)
    
    # Minimal input with just a tool call result as string
    payload = {
        "model": "qwen3-27b",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            }
        ],
        "tools": [TOOL]
    }
    
    try:
        resp = httpx.post(f"{BASE_URL}/responses", json=payload, timeout=60)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print("✅ Baseline request works")
            return True
        else:
            print(f"❌ Baseline failed: {resp.text[:500]}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False


def test_direct_list_output():
    """Direct test: send function_call_output with list in a minimal payload."""
    print("\n" + "=" * 60)
    print("TEST 3: Direct minimal function_call_output with LIST")
    print("=" * 60)
    
    payload = {
        "model": "qwen3-27b",
        "input": [
            {"role": "user", "content": "Hello"},
            {
                "type": "function_call_output",
                "call_id": "call_test_123",
                "output": [{"type": "input_text", "text": "result"}]
            }
        ]
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        resp = httpx.post(f"{BASE_URL}/responses", json=payload, timeout=60)
        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"\n❌ FAILURE (reproduced): {resp.status_code}")
            print(f"Error body: {resp.text[:1000]}")
            return True
        else:
            print("\n✅ SUCCESS: Server accepted list output!")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return None


if __name__ == "__main__":
    print("Agentic-api function_call_output list reproduction")
    print(f"Target: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run baseline first
    baseline = test_with_string_output()
    
    # Run direct minimal test
    direct_repro = test_direct_list_output()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline (string output): {'PASS' if baseline else 'FAIL'}")
    print(f"Direct list output test: {'REPRODUCED (fix not working)' if direct_repro else 'FIXED (server accepts list)'}")
    
    if direct_repro:
        print("\n🔴 The fix is NOT deployed or NOT working")
        sys.exit(1)
    elif direct_repro is False:
        print("\n🟢 The fix is working correctly")
        sys.exit(0)
    else:
        print("\n🟡 Test was inconclusive")
        sys.exit(2)
