#!/usr/bin/env python3
"""
Simple CLI Chat Application
A production-ready command-line chat interface for llama.cpp models.

Features:
- Conversation history management
- System prompt configuration
- Multi-turn conversations
- Save/load chat sessions
- Configurable model parameters
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import yaml


class ChatHistory:
    """Manages conversation history and persistence."""

    def __init__(self, session_name: Optional[str] = None):
        self.history_dir = Path.home() / ".llama_chat" / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

        if session_name:
            self.session_file = self.history_dir / f"{session_name}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_file = self.history_dir / f"session_{timestamp}.json"

        self.messages: List[Dict[str, str]] = []
        self.load_session()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the conversation."""
        return self.messages

    def get_context(self) -> str:
        """Get formatted conversation context for the model."""
        context = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                context.append(f"System: {content}")
            elif role == "user":
                context.append(f"User: {content}")
            elif role == "assistant":
                context.append(f"Assistant: {content}")
        return "\n".join(context)

    def save_session(self):
        """Save conversation history to disk."""
        data = {
            "session_file": str(self.session_file),
            "created_at": self.messages[0]["timestamp"] if self.messages else datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": len(self.messages),
            "messages": self.messages
        }

        with open(self.session_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n[Session saved to {self.session_file}]", file=sys.stderr)

    def load_session(self):
        """Load conversation history from disk."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    self.messages = data.get("messages", [])
                print(f"\n[Loaded session from {self.session_file}]", file=sys.stderr)
                print(f"[Messages in history: {len(self.messages)}]", file=sys.stderr)
            except json.JSONDecodeError:
                print(f"\n[Warning: Could not load session file, starting fresh]", file=sys.stderr)

    def list_sessions(self) -> List[Path]:
        """List all saved sessions."""
        return sorted(self.history_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        print("\n[History cleared]", file=sys.stderr)


class Config:
    """Manages application configuration."""

    DEFAULT_CONFIG = {
        "model": {
            "path": "./models/llama-2-7b-chat.Q4_K_M.gguf",
            "context_size": 2048,
            "threads": 4,
            "gpu_layers": 0
        },
        "generation": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "max_tokens": 512
        },
        "system_prompts": {
            "default": "You are a helpful AI assistant.",
            "coding": "You are an expert programmer. Provide clear, concise code examples and explanations.",
            "creative": "You are a creative writing assistant. Help with storytelling and creative content.",
            "technical": "You are a technical expert. Provide detailed, accurate technical information."
        },
        "llama_cpp_path": "./llama-cli"
    }

    def __init__(self, config_file: Optional[str] = None):
        self.config_dir = Path.home() / ".llama_chat"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        if config_file:
            self.config_file = Path(config_file)
        else:
            self.config_file = self.config_dir / "config.yaml"

        self.config = self.load_config()

    def load_config(self) -> dict:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    # Merge with defaults to handle missing keys
                    return self._merge_configs(self.DEFAULT_CONFIG, config)
            except yaml.YAMLError as e:
                print(f"\n[Warning: Could not load config: {e}]", file=sys.stderr)
                print("[Using default configuration]", file=sys.stderr)
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config file
            self.save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()

    def _merge_configs(self, default: dict, custom: dict) -> dict:
        """Recursively merge custom config with defaults."""
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def save_config(self, config: dict):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\n[Configuration saved to {self.config_file}]", file=sys.stderr)

    def get_system_prompt(self, prompt_name: str) -> str:
        """Get a system prompt by name."""
        return self.config["system_prompts"].get(prompt_name,
                                                  self.config["system_prompts"]["default"])


class LlamaChatClient:
    """Client for interacting with llama.cpp."""

    def __init__(self, config: Config):
        self.config = config
        self.llama_path = config.config["llama_cpp_path"]

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the model."""
        model_config = self.config.config["model"]
        gen_config = self.config.config["generation"]

        # Build the full prompt with system message if provided
        full_prompt = ""
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\n{prompt}\n\nAssistant:"
        else:
            full_prompt = f"{prompt}\n\nAssistant:"

        # Build llama.cpp command
        cmd = [
            self.llama_path,
            "-m", model_config["path"],
            "-c", str(model_config["context_size"]),
            "-t", str(model_config["threads"]),
            "-ngl", str(model_config["gpu_layers"]),
            "--temp", str(gen_config["temperature"]),
            "--top-p", str(gen_config["top_p"]),
            "--top-k", str(gen_config["top_k"]),
            "--repeat-penalty", str(gen_config["repeat_penalty"]),
            "-n", str(gen_config["max_tokens"]),
            "-p", full_prompt,
            "--no-display-prompt"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"\n[Error running llama.cpp: {e}]", file=sys.stderr)
            print(f"[stderr: {e.stderr}]", file=sys.stderr)
            raise
        except FileNotFoundError:
            print(f"\n[Error: llama.cpp not found at {self.llama_path}]", file=sys.stderr)
            print("[Please update the 'llama_cpp_path' in your config file]", file=sys.stderr)
            raise


def print_welcome():
    """Print welcome message."""
    print("=" * 60)
    print("  Simple CLI Chat - llama.cpp Interface")
    print("=" * 60)
    print("\nCommands:")
    print("  /help     - Show this help message")
    print("  /save     - Save current session")
    print("  /clear    - Clear conversation history")
    print("  /sessions - List all saved sessions")
    print("  /quit     - Exit the chat")
    print("\nType your message and press Enter to chat.")
    print("=" * 60)


def print_sessions(history: ChatHistory):
    """Print list of saved sessions."""
    sessions = history.list_sessions()
    if not sessions:
        print("\n[No saved sessions found]")
        return

    print("\n" + "=" * 60)
    print("Saved Sessions:")
    print("=" * 60)
    for i, session in enumerate(sessions[:10], 1):  # Show last 10
        try:
            with open(session, 'r') as f:
                data = json.load(f)
                msg_count = data.get("message_count", 0)
                updated = data.get("updated_at", "Unknown")
                print(f"{i}. {session.stem}")
                print(f"   Messages: {msg_count}, Last updated: {updated[:19]}")
        except:
            print(f"{i}. {session.stem}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Simple CLI Chat Application for llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Start new chat session
  %(prog)s --session my-chat                  # Load/create named session
  %(prog)s --system-prompt coding             # Use coding system prompt
  %(prog)s --config my-config.yaml            # Use custom config file
  %(prog)s --model ./models/my-model.gguf     # Override model path
        """
    )

    parser.add_argument(
        "--session", "-s",
        help="Session name to load or create"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--system-prompt", "-p",
        default="default",
        help="System prompt to use (default, coding, creative, technical)"
    )
    parser.add_argument(
        "--model", "-m",
        help="Override model path from config"
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't load conversation history"
    )

    args = parser.parse_args()

    # Initialize components
    config = Config(args.config)

    # Override model if specified
    if args.model:
        config.config["model"]["path"] = args.model

    history = ChatHistory(args.session)
    if args.no_history:
        history.clear_history()

    client = LlamaChatClient(config)

    # Get system prompt
    system_prompt = config.get_system_prompt(args.system_prompt)
    if not history.messages or history.messages[0]["role"] != "system":
        history.add_message("system", system_prompt)

    # Print welcome
    print_welcome()
    print(f"\n[Using system prompt: {args.system_prompt}]")
    print(f"[Model: {config.config['model']['path']}]")
    print(f"[Session: {history.session_file.stem}]\n")

    # Main chat loop
    try:
        while True:
            try:
                user_input = input("\n\033[1;34mYou:\033[0m ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input[1:].lower()

                    if command == "quit" or command == "exit":
                        history.save_session()
                        print("\n[Goodbye!]\n")
                        break

                    elif command == "help":
                        print_welcome()
                        continue

                    elif command == "save":
                        history.save_session()
                        continue

                    elif command == "clear":
                        history.clear_history()
                        history.add_message("system", system_prompt)
                        continue

                    elif command == "sessions":
                        print_sessions(history)
                        continue

                    else:
                        print(f"\n[Unknown command: {command}]")
                        continue

                # Add user message to history
                history.add_message("user", user_input)

                # Generate response
                print("\n\033[1;32mAssistant:\033[0m ", end="", flush=True)

                context = history.get_context()
                response = client.generate_response(context)

                print(response)

                # Add assistant response to history
                history.add_message("assistant", response)

            except KeyboardInterrupt:
                print("\n\n[Use /quit to exit]")
                continue
            except EOFError:
                history.save_session()
                print("\n\n[Goodbye!]\n")
                break
            except Exception as e:
                print(f"\n\n[Error: {e}]", file=sys.stderr)
                print("[Continuing...]\n", file=sys.stderr)
                continue

    except Exception as e:
        print(f"\n[Fatal error: {e}]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
