# FrameForge Studio Voice Command Integration

This tool integrates Whisper.cpp for speech-to-text and Llama.cpp for intent classification to provide voice command functionality for FrameForge Studio.

## Overview

The FrameForge Sidecar is a 64-bit resident process that:
1. Receives audio input (via file or IPC)
2. Transcribes audio to text using Whisper
3. Classifies intent and extracts parameters using Llama
4. Validates commands against a strict schema
5. Sends validated JSON commands to the 32-bit FrameForge Bridge via Named Pipes

## Command Schema

Commands follow a JSON-based schema with the following structure:

```json
{
  "verb": "PAN",
  "subject": "Camera1",
  "action_group": "CAMERA_CONTROL",
  "parameters": {
    "direction": "LEFT",
    "degrees": 45.0,
    "speed": 10.0,
    "target": "ObjectName",
    "pose_description": "arms raised above head",
    "joint_rotations": [
      {"name": "shoulder_left", "rotation_x": 0, "rotation_y": 90, "rotation_z": 0}
    ]
  }
}
```

### Action Groups

- **CAMERA_CONTROL**: Camera movements (PAN, TILT, DOLLY, ZOOM, LEAN)
- **ACTOR_POSE**: Actor positioning (SET_POSE, ADJUST_POSE)
- **OBJECT_MGMT**: Object manipulation (ADD, DELETE, MOVE, ROTATE)
- **SHOT_MGMT**: Shot management (SHOT, SAVE_SHOT, LOAD_SHOT)

### Verbs and Required Parameters

| Verb | Required Parameters |
|------|---------------------|
| PAN | direction |
| TILT | direction |
| DOLLY | direction, speed |
| ZOOM | direction |
| LEAN | direction, degrees |
| SET_POSE | pose_description |
| ADJUST_POSE | pose_description |
| ADD | target |
| DELETE | target |
| MOVE | target, direction |
| ROTATE | target, degrees |
| SHOT | target |
| SAVE_SHOT | target |
| LOAD_SHOT | target |

## Building

The tool is built as part of the llama.cpp build process:

```bash
cmake -B build
cmake --build build --config Release
```

The binary will be located at: `build/bin/frameforge-sidecar`

## Usage

### Test Mode (with audio file)

```bash
./build/bin/frameforge-sidecar \
  --whisper-model /path/to/whisper-model.bin \
  --llama-model /path/to/llama-model.gguf \
  --audio /path/to/audio.wav \
  --verbose
```

### Server Mode (IPC with Named Pipes)

```bash
./build/bin/frameforge-sidecar \
  --whisper-model /path/to/whisper-model.bin \
  --llama-model /path/to/llama-model.gguf \
  --pipe frameforge_pipe
```

### Command-Line Options

- `-wm, --whisper-model FNAME` - Path to Whisper model file (required)
- `-lm, --llama-model FNAME` - Path to Llama model file (required)
- `-a, --audio FILE` - Audio file to transcribe (for testing)
- `-p, --pipe NAME` - Named pipe name (default: frameforge_pipe)
- `-vd, --verb-defs FILE` - Path to verb definitions JSON file (optional)
- `-t, --threads N` - Number of threads (default: 4)
- `-v, --verbose` - Enable verbose output
- `-h, --help` - Show help message

### Verb Definitions

As of the latest version, verb definitions can be loaded from a JSON file instead of being hard-coded. This allows for easier customization and extension without modifying the source code.

**JSON File Format:**

The verb definitions file should follow this structure:

```json
{
  "action_groups": {
    "CAMERA_CONTROL": "Camera movements and controls",
    "ACTOR_POSE": "Actor positioning and poses",
    "OBJECT_MGMT": "Object manipulation",
    "SHOT_MGMT": "Shot management"
  },
  "verbs": [
    {
      "name": "PAN",
      "action_group": "CAMERA_CONTROL",
      "required_parameters": ["direction"],
      "aliases": ["PIN"],
      "description": "Pan the camera left or right"
    }
  ]
}
```

**Using Custom Verb Definitions:**

```bash
./build/bin/frameforge-sidecar \
  -wm whisper-base.en.bin \
  -lm llama-3-8b-instruct.gguf \
  -vd /path/to/custom-verbs.json \
  -a test_command.wav \
  -v
```

A sample verb definitions file is provided at `tools/frameforge/verb-definitions.json` containing all the currently defined verbs.

If no verb definitions file is specified, the system will fall back to hard-coded defaults.

## Architecture

### Components

1. **frameforge-schema**: Defines the command schema, action groups, verbs, and parameters
2. **frameforge-validator**: CommandValidator class for validating commands
3. **frameforge-json**: JSON serialization/deserialization utilities
4. **frameforge-ipc**: IPC server/client using Named Pipes
5. **frameforge-sidecar**: Main application integrating Whisper and Llama

### Intent Classification

The system uses a strict system prompt for Llama to classify intents:

- Maps natural language to specific verbs
- Handles common misspellings (e.g., "PIN LEFT" → "PAN LEFT")
- Extracts parameters from context
- Generates joint rotation arrays for pose descriptions
- Returns only valid JSON

### Validation

The CommandValidator checks:
- Verb is valid and recognized
- Action group matches the verb
- All required parameters are present
- Parameter values are within valid ranges
- Subject is not empty

If validation fails, the system generates a clarification request asking the user for missing information.

### IPC Communication

Named Pipes provide high-speed communication:
- **Windows**: `\\.\pipe\frameforge_pipe`
- **Unix/Linux**: `/tmp/frameforge_pipe`

Messages are length-prefixed (4-byte size + payload) for reliable streaming.

## Example Voice Commands

- "Pan the camera left"
- "Tilt camera 1 up 30 degrees"
- "Add a chair to the scene"
- "Set Tom's pose to arms crossed"
- "Move the table forward"
- "Save shot as establishing"

## Models

### Recommended Models

**Whisper**:
- `whisper-base.en` - Fast, English-only
- `whisper-small.en` - Better accuracy
- Download from: https://huggingface.co/ggerganov/whisper.cpp

**Llama**:
- `llama-3-8b-instruct.gguf` - Good balance of speed and accuracy
- `llama-3.1-8b-instruct.gguf` - Latest version with better instruction following
- Download from: https://huggingface.co/models

## Development

### Adding New Verbs

1. Add the verb to the `Verb` enum in `frameforge-schema.h`
2. Update `string_to_verb()` in `frameforge-schema.cpp`
3. Add to the appropriate action group in `get_action_group_for_verb()`
4. Define required parameters in `get_required_parameters()`

### Testing

Create test audio files with commands and run in test mode:

```bash
./build/bin/frameforge-sidecar \
  -wm whisper-base.en.bin \
  -lm llama-3-8b-instruct.gguf \
  -a test_command.wav \
  -v
```

## License

This tool is part of llama.cpp and follows the same MIT license.
