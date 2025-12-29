# Semantic AI Server for FrameForge Studio

A 64-bit C++ Semantic AI Server that processes natural language commands for FrameForge Studio using llama.cpp for intent classification and command validation.

## Overview

This server implements:

1. **JSON Command Schema**: Defines verbs (PAN, TILT, LEAN, ROLL, ADD, DELETE, SHOT, etc.) with required and optional parameters
2. **Intent Engine**: Uses Llama-3 (or similar models) with a strict system prompt for intent classification
3. **Command Validation**: Validates LLM output against command schemas
4. **IPC Communication**: Named Pipes for high-speed inter-process communication with external applications

## Architecture

### Components

- **CommandSchema**: Defines action groups, verbs, and parameter specifications
- **CommandValidator**: Validates JSON commands against schemas with fuzzy verb matching
- **IntentEngine**: Processes text input through Llama for classification and command generation
- **IPCHandler**: Named Pipes implementation for Windows and Unix systems

### Action Groups

- **CAMERA_CONTROL**: Camera movements (PAN, TILT, LEAN, ROLL, DOLLY, TRUCK, PEDESTAL, ZOOM, FOCUS)
- **ACTOR_POSE**: Actor pose modifications (MODIFY)
- **OBJECT_MGMT**: Scene object management (ADD, DELETE, SELECT)
- **SHOT_MGMT**: Shot creation and editing (SHOT, CUT)

## Building

The semantic server is built as part of the llama.cpp build system:

```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

The executable will be located at: `build/bin/llama-semantic-server`

## Usage

### Interactive Mode

For testing and development:

```bash
./build/bin/llama-semantic-server -m llama-3-8b-instruct.gguf --interactive
```

Enter natural language commands:

```
> Pan left 30 degrees
Result: {
  "verb": "PAN",
  "action_group": "CAMERA_CONTROL",
  "parameters": {
    "direction": "LEFT",
    "degrees": 30.0,
    "speed": 1.0
  }
}

> Add a chair
Result: {
  "verb": "ADD",
  "action_group": "OBJECT_MGMT",
  "parameters": {
    "object_type": "chair"
  }
}
```

### IPC Mode (Named Pipes)

For production use with external applications:

```bash
./build/bin/llama-semantic-server -m llama-3-8b-instruct.gguf --pipe-name frameforge_semantic
```

The server will:
1. Create a named pipe (`frameforge_semantic` on Unix, `\\.\pipe\frameforge_semantic` on Windows)
2. Listen for incoming messages
3. Process text through the Intent Engine
4. Return validated JSON commands

### Integration with Whisper

While whisper.cpp is a separate repository, you can integrate audio transcription by:

1. Running whisper.cpp to transcribe audio to text
2. Sending the transcribed text to the semantic server via named pipe
3. The semantic server will classify and validate commands
4. Validated JSON commands are sent back through the pipe

Example integration:

```json
// Input to semantic server (from whisper or other source):
{"text": "pan left thirty degrees"}

// Output from semantic server:
{
  "verb": "PAN",
  "action_group": "CAMERA_CONTROL",
  "parameters": {
    "direction": "LEFT",
    "degrees": 30.0,
    "speed": 1.0
  }
}
```

## Command Schema Examples

### Camera Control

```json
// Pan command
{
  "verb": "PAN",
  "action_group": "CAMERA_CONTROL",
  "parameters": {
    "direction": "LEFT",  // Required: "LEFT" or "RIGHT"
    "degrees": 45.0,      // Optional: rotation amount
    "speed": 0.5          // Optional: movement speed
  }
}

// Dolly command
{
  "verb": "DOLLY",
  "action_group": "CAMERA_CONTROL",
  "parameters": {
    "direction": "IN",    // Required: "IN" or "OUT"
    "distance": 10.0,     // Optional: distance to move
    "speed": 1.0          // Optional: movement speed
  }
}
```

### Actor Pose

```json
{
  "verb": "MODIFY",
  "action_group": "ACTOR_POSE",
  "parameters": {
    "subject": "actor",          // Required: which actor
    "pose_description": "sitting", // Required: pose description
    "joint_rotations": []         // Optional: specific joint angles
  }
}
```

### Object Management

```json
// Add object
{
  "verb": "ADD",
  "action_group": "OBJECT_MGMT",
  "parameters": {
    "object_type": "chair",  // Required: type of object
    "name": "chair_01",      // Optional: object name
    "position": {},          // Optional: initial position
    "properties": {}         // Optional: additional properties
  }
}

// Delete object
{
  "verb": "DELETE",
  "action_group": "OBJECT_MGMT",
  "parameters": {
    "target": "chair_01"  // Required: object to delete
  }
}
```

### Shot Management

```json
{
  "verb": "SHOT",
  "action_group": "SHOT_MGMT",
  "parameters": {
    "shot_type": "CLOSE",  // Required: "WIDE", "MEDIUM", "CLOSE", "ECU"
    "subject": "actor",    // Optional: shot subject
    "duration": 5.0        // Optional: shot duration
  }
}
```

## Error Handling

When required parameters are missing, the server returns an error:

```json
{
  "error": true,
  "message": "Missing required parameters for PAN: direction",
  "missing_parameters": ["direction"]
}
```

## Fuzzy Matching

The CommandValidator includes fuzzy matching for typo correction:

- "PIN LEFT" → "PAN LEFT"
- "TILE UP" → "TILT UP"
- Levenshtein distance ≤ 2 for suggestions

## System Requirements

- 64-bit operating system (Windows, Linux, macOS)
- 8GB+ RAM (depends on model size)
- Llama-3 or compatible instruction-tuned model

## Recommended Models

- **llama-3-8b-instruct**: Good balance of speed and accuracy
- **llama-3-70b-instruct**: Higher accuracy, requires more resources
- **mistral-7b-instruct**: Faster alternative

## Notes

- The server runs as a resident process (sidecar architecture)
- Named pipes provide low-latency IPC communication
- System prompt is optimized for FrameForge Studio commands
- All commands are validated against the schema before being sent to the client
- The temperature is set low (0.1) for more deterministic classification

## Future Enhancements

- Support for batch command processing
- Command history and undo/redo
- Multi-language support
- WebSocket alternative to named pipes
- Direct whisper.cpp integration when available
