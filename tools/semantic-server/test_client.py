#!/usr/bin/env python3
"""
Test client for the Semantic AI Server
Demonstrates Named Pipe communication
"""

import sys
import json
import time
import platform

def send_to_pipe(pipe_name, message):
    """Send a message through the named pipe and receive response"""
    
    if platform.system() == 'Windows':
        import win32file
        import win32pipe
        import pywintypes
        
        pipe_path = f"\\\\.\\pipe\\{pipe_name}"
        
        try:
            # Connect to the pipe
            handle = win32file.CreateFile(
                pipe_path,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None
            )
            
            # Send message
            win32file.WriteFile(handle, message.encode())
            
            # Read response
            result = win32file.ReadFile(handle, 4096)
            response = result[1].decode()
            
            win32file.CloseHandle(handle)
            
            return response
            
        except pywintypes.error as e:
            print(f"Error communicating with pipe: {e}")
            return None
    else:
        # Unix/Linux
        pipe_path = pipe_name
        
        try:
            # Open pipe for writing
            with open(pipe_path, 'w') as pipe_out:
                pipe_out.write(message)
                pipe_out.flush()
            
            # Open pipe for reading (response)
            # Note: In a real implementation, you'd need a response pipe
            # For now, we just send the command
            print("Command sent to pipe (response mechanism depends on your setup)")
            return "Command sent"
            
        except Exception as e:
            print(f"Error communicating with pipe: {e}")
            return None

def main():
    if len(sys.argv) < 2:
        print("Usage: test_client.py <command> [pipe_name]")
        print("\nExamples:")
        print("  python3 test_client.py 'pan left 30 degrees'")
        print("  python3 test_client.py 'add a chair' frameforge_semantic")
        sys.exit(1)
    
    command = sys.argv[1]
    pipe_name = sys.argv[2] if len(sys.argv) > 2 else "frameforge_semantic"
    
    print(f"Sending command to semantic server: {command}")
    print(f"Pipe name: {pipe_name}")
    print("-" * 60)
    
    # Send plain text command
    response = send_to_pipe(pipe_name, command)
    
    if response:
        try:
            # Try to parse as JSON and pretty print
            json_response = json.loads(response)
            print("\nResponse:")
            print(json.dumps(json_response, indent=2))
            
            if json_response.get('error'):
                print("\n⚠️  Command validation failed!")
            else:
                print("\n✓ Command validated successfully")
                
        except json.JSONDecodeError:
            print(f"\nResponse (raw): {response}")
    else:
        print("\n❌ No response received")
        print("\nMake sure the semantic server is running:")
        print(f"  ./build/bin/llama-semantic-server -m <model.gguf> --pipe-name {pipe_name}")

if __name__ == "__main__":
    main()
