"""
Module: 02-basic-chat.py
Purpose: Demonstrate interactive chat with context management
Learning Objectives:
    - Build a simple interactive chat loop
    - Manage conversation history and context
    - Handle user input and exit conditions
    - Understand token counting and context limits

Prerequisites: Module 1 Lesson 1.4 - First Inference complete
Estimated Time: 20 minutes
Module: 1.4 - Basic Inference
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Install it with: pip install llama-cpp-python")
    sys.exit(1)


class ChatBot:
    """
    A simple chatbot that maintains conversation context.

    This class demonstrates how to:
    - Manage conversation history
    - Build prompts with context
    - Handle multi-turn conversations
    - Respect context window limits
    """

    def __init__(
        self,
        model: Llama,
        system_prompt: Optional[str] = None,
        max_history: int = 10
    ):
        """
        Initialize the chatbot.

        Args:
            model: Loaded Llama model instance
            system_prompt: System message to set chatbot behavior
            max_history: Maximum number of conversation turns to keep in context
        """
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Args:
            role: Either "user" or "assistant"
            content: The message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })

        # Keep only the most recent messages to avoid context overflow
        if len(self.conversation_history) > self.max_history * 2:
            # Remove oldest user-assistant pair
            self.conversation_history = self.conversation_history[2:]

    def build_prompt(self) -> str:
        """
        Build a prompt string from conversation history.

        This demonstrates one way to format multi-turn conversations.
        Different models may prefer different formats (ChatML, Alpaca, etc.).

        Returns:
            Formatted prompt string ready for model input
        """
        # Start with system prompt
        prompt_parts = [f"System: {self.system_prompt}\n"]

        # Add conversation history
        for msg in self.conversation_history:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}\n")

        # Add assistant prefix to prompt completion
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def chat(self, user_message: str, max_tokens: int = 256) -> str:
        """
        Process a user message and generate a response.

        Args:
            user_message: The user's input message
            max_tokens: Maximum tokens to generate in response

        Returns:
            The assistant's response
        """
        # Add user message to history
        self.add_message("user", user_message)

        # Build the full prompt with context
        prompt = self.build_prompt()

        # Generate response
        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["User:", "System:", "\n\n"],  # Stop at next turn or double newline
                echo=False
            )

            response = output["choices"][0]["text"].strip()

            # Add assistant response to history
            self.add_message("assistant", response)

            return response

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(error_msg, file=sys.stderr)
            return "[Error: Failed to generate response]"

    def reset(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []

    def get_history_summary(self) -> str:
        """Get a summary of the conversation history."""
        return f"Conversation has {len(self.conversation_history)} messages"


def load_model(model_path: str, n_ctx: int = 2048) -> Optional[Llama]:
    """
    Load a GGUF model for chat.

    Args:
        model_path: Path to .gguf model file
        n_ctx: Context window size

    Returns:
        Loaded model or None if loading fails
    """
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return None

    try:
        print(f"Loading model: {model_path}")
        print("Please wait...")

        llm = Llama(
            model_path=str(model_file),
            n_ctx=n_ctx,
            n_gpu_layers=0,  # Set to -1 for GPU acceleration
            verbose=False
        )

        print(f"✓ Model loaded (context: {llm.n_ctx()} tokens)\n")
        return llm

    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None


def print_welcome() -> None:
    """Print welcome message and instructions."""
    print("=" * 70)
    print("LLaMA.cpp Python - Basic Chat Example")
    print("=" * 70)
    print()
    print("This example demonstrates an interactive chat loop with context.")
    print()
    print("Commands:")
    print("  Type your message and press Enter to chat")
    print("  'exit' or 'quit' - Exit the chat")
    print("  'clear' or 'reset' - Clear conversation history")
    print("  'history' - Show conversation summary")
    print("  'help' - Show this help message")
    print()
    print("=" * 70)
    print()


def print_help() -> None:
    """Print help message."""
    print("\nAvailable commands:")
    print("  exit, quit    - Exit the chat")
    print("  clear, reset  - Clear conversation history")
    print("  history       - Show conversation summary")
    print("  help          - Show this help message")
    print()


def main() -> None:
    """
    Main chat loop demonstration.

    This example shows:
    1. Interactive chat interface
    2. Context management across turns
    3. Command handling (exit, reset, etc.)
    4. Conversation history tracking
    """
    # Get model path from command line or use default
    model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Print welcome message
    print_welcome()

    # Load the model
    model = load_model(model_path, n_ctx=2048)
    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)

    # Create chatbot with a friendly system prompt
    system_prompt = (
        "You are a helpful and friendly AI assistant. "
        "Provide concise and informative responses. "
        "Be conversational and helpful."
    )

    chatbot = ChatBot(
        model=model,
        system_prompt=system_prompt,
        max_history=10  # Keep last 10 exchanges
    )

    print("Chat started! Type your message or 'help' for commands.\n")

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye! Thanks for chatting.")
                break

            elif user_input.lower() in ["clear", "reset"]:
                chatbot.reset()
                print("\n✓ Conversation history cleared.\n")
                continue

            elif user_input.lower() == "history":
                print(f"\n{chatbot.get_history_summary()}\n")
                continue

            elif user_input.lower() == "help":
                print_help()
                continue

            # Generate and display response
            print("Assistant: ", end="", flush=True)
            response = chatbot.chat(user_input, max_tokens=256)
            print(response)
            print()  # Empty line for readability

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break

        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            print("Continuing...\n")
            continue

    print("\n" + "=" * 70)
    print("Chat session ended.")
    print(f"Final status: {chatbot.get_history_summary()}")
    print("=" * 70)


if __name__ == "__main__":
    # Example usage documentation
    """
    Usage:
        # Run with default model
        python 02-basic-chat.py

        # Run with custom model
        python 02-basic-chat.py path/to/model.gguf

    Example session:
        You: Hello!
        Assistant: Hello! How can I help you today?

        You: What's the capital of France?
        Assistant: The capital of France is Paris.

        You: Tell me more about it
        Assistant: Paris is a beautiful city located in northern France...
        [The bot remembers we're talking about Paris]

        You: clear
        ✓ Conversation history cleared.

        You: exit
        Goodbye! Thanks for chatting.

    Key Concepts Demonstrated:
    1. Interactive input loop with input()
    2. Conversation context management
    3. Command handling (exit, clear, help)
    4. Error handling for interrupted sessions
    5. Building prompts with history
    6. Token management within context window
    """
    main()
