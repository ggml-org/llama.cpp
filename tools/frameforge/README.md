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

Commands follow a JSON-based schema optimized for Delphi Bridge compatibility:

```json
{
  "verb": "PAN",
  "master_verb": "START",
  "action_group": "CAMERA_CONTROL",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "parameters": {
    "subject": "Camera1",
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

**Key Changes from Previous Version:**
- `subject` field moved into `parameters` object
- Added `timestamp` field (ISO 8601 format)
- Added `master_verb` field for compound commands
- Removed `subject` from root level

### Action Groups

- **CAMERA_CONTROL**: Camera movements (PAN, TILT, DOLLY, ZOOM, LEAN)
- **ACTOR_POSE**: Actor positioning (SET_POSE, ADJUST_POSE)
- **OBJECT_MGMT**: Object manipulation (ADD, DELETE, MOVE, ROTATE)
- **SHOT_MGMT**: Shot management (SHOT, SAVE_SHOT, LOAD_SHOT)
- **MASTER_VERB**: Master verbs requiring secondary verbs (START, BEGIN, HAVE, MAKE, STOP)

### Master Verbs

Master verbs are special verbs that require a secondary verb to form complete commands:

- **START/BEGIN**: Initiates an action (e.g., "START PANNING LEFT")
- **HAVE/MAKE**: Commands an actor/object (e.g., "HAVE TOM WALK TO THE DOOR")
- **STOP**: Stops an ongoing action (e.g., "STOP PANNING")

Example with master verb:
```json
{
  "verb": "PAN",
  "master_verb": "START",
  "action_group": "CAMERA_CONTROL",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "parameters": {
    "direction": "LEFT",
    "speed": 5.0
  }
}
```

### Verbs and Parameters

Each verb has **required parameters** and **optional parameters**:

| Verb | Required Parameters | Optional Parameters |
|------|---------------------|---------------------|
| START | - | subject, target, direction, speed, degrees |
| BEGIN | - | subject, target, direction, speed, degrees |
| HAVE | subject | target, direction, speed, degrees, pose_description |
| MAKE | subject | target, direction, speed, degrees, pose_description |
| STOP | - | subject, target |
| PAN | direction | speed, degrees, target, subject |
| TILT | direction | speed, degrees, target, subject |
| DOLLY | direction | speed, target, subject |
| ZOOM | direction | speed, degrees, target, subject |
| LEAN | direction, degrees | speed, target, subject |
| SET_POSE | subject, pose_description | target |
| ADJUST_POSE | subject, pose_description | target |
| ADD | target | subject |
| DELETE | target | subject |
| MOVE | target, direction | speed, degrees, subject |
| ROTATE | target, degrees | direction, speed, subject |
| SHOT | target | subject |
| SAVE_SHOT | target | subject |
| LOAD_SHOT | target | subject |

### Verb Disambiguation

The system uses parameter patterns to help identify misrecognized verbs. For example:
- "ROOM LEFT" doesn't match ZOOM pattern (requires direction like IN/OUT)
- "ROOM OUT" matches ZOOM pattern → corrected to "ZOOM OUT"

This helps correct common speech recognition errors.

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

The verb definitions file supports the following structure:

```json
{
  "action_groups": {
    "CAMERA_CONTROL": "Camera movements and controls",
    "ACTOR_POSE": "Actor positioning and poses",
    "OBJECT_MGMT": "Object manipulation",
    "SHOT_MGMT": "Shot management",
    "MASTER_VERB": "Master verbs that require secondary verbs"
  },
  "verbs": [
    {
      "name": "PAN",
      "action_group": "CAMERA_CONTROL",
      "required_parameters": ["direction"],
      "optional_parameters": ["speed", "degrees", "target", "subject"],
      "aliases": ["PIN"],
      "is_master_verb": false,
      "description": "Pan the camera left or right"
    },
    {
      "name": "START",
      "action_group": "MASTER_VERB",
      "required_parameters": [],
      "optional_parameters": ["subject", "target", "direction", "speed", "degrees"],
      "aliases": ["BEGIN"],
      "is_master_verb": true,
      "description": "Begin an action (requires secondary verb)"
    }
  ]
}
```

**Fields:**
- `name`: The verb identifier (must match enum in code)
- `action_group`: Category this verb belongs to
- `required_parameters`: Parameters that must be present for validation
- `optional_parameters`: Parameters that may be present (used for verb disambiguation)
- `aliases`: Alternative names that map to this verb (e.g., "PIN" → "PAN", "ROOM" → "ZOOM")
- `is_master_verb`: Boolean indicating if this is a master verb
- `description`: Human-readable description

**Verb Disambiguation:**
Optional parameters help identify misrecognized verbs. For example:
- If Whisper hears "ROOM LEFT", the system checks if LEFT is valid for ZOOM
- Since ZOOM expects IN/OUT (not LEFT), it's likely incorrect
- If it hears "ROOM OUT", OUT matches ZOOM's direction pattern → corrected to "ZOOM OUT"

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

**Camera Control:**
- "Pan the camera left"
- "Start panning slowly to the right"
- "Tilt camera 1 up 30 degrees"
- "Begin zooming in"

**Object Management:**
- "Add a chair to the scene"
- "Move the table forward"
- "Have Tom walk to the door"
- "Make Rachel turn around"

**Actor Pose:**
- "Set Tom's pose to arms crossed"
- "Have Sarah raise her hands"
- "Make Tom slap Rachel"

**Shot Management:**
- "Save shot as establishing"
- "Load the closeup shot"

**Master Verb Examples:**
- "Start panning slowly left" → START + PAN with direction=LEFT, speed modifier
- "Begin pushing in" → BEGIN + DOLLY with direction=FORWARD
- "Have Tom walk to the door" → HAVE + MOVE with subject=Tom, target=door
- "Make Rachel slap Tom" → MAKE + custom action with subject=Rachel, target=Tom
- "Stop" → STOP (stops current action)

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
