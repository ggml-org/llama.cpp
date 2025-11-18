#!/usr/bin/env python3
"""
Simple Chat Application using llama-cpp-python

This example demonstrates:
- Basic chat interface
- Conversation history management
- Streaming responses
- System prompts
"""

from llama_cpp import Llama
from typing import List, Dict
import sys


class SimpleChatApp:
    """Simple command-line chat application."""

    def __init__(
        self,
        model_path: str,
        system_prompt: str = "You are a helpful AI assistant.",
        n_ctx: int = 2048,
        n_gpu_layers: int = 0
    ):
        """
        Initialize chat application.

        Args:
            model_path: Path to GGUF model file
            system_prompt: System instruction for the model
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
        """
        print("Loading model...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format="llama-2",
            verbose=False
        )

        self.conversation: List[Dict[str, str]] = []
        if system_prompt:
            self.conversation.append({
                "role": "system",
                "content": system_prompt
            })

        print("Model loaded successfully!")

    def chat(self, user_message: str, stream: bool = True) -> str:
        """
        Send a message and get response.

        Args:
            user_message: User's message
            stream: Whether to stream the response

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation.append({
            "role": "user",
            "content": user_message
        })

        if stream:
            # Streaming response
            print("\nAssistant: ", end="", flush=True)
            full_response = ""

            stream = self.llm.create_chat_completion(
                messages=self.conversation,
                max_tokens=512,
                temperature=0.7,
                stream=True
            )

            for chunk in stream:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    content = delta['content']
                    print(content, end="", flush=True)
                    full_response += content

            print()  # Newline after response

            # Add to history
            self.conversation.append({
                "role": "assistant",
                "content": full_response
            })

            return full_response
        else:
            # Non-streaming response
            response = self.llm.create_chat_completion(
                messages=self.conversation,
                max_tokens=512,
                temperature=0.7
            )

            assistant_message = response['choices'][0]['message']['content']

            # Add to history
            self.conversation.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

    def clear_history(self):
        """Clear conversation history (keep system prompt)."""
        system_msgs = [msg for msg in self.conversation if msg['role'] == 'system']
        self.conversation = system_msgs

    def run(self):
        """Run interactive chat loop."""
        print("\n" + "=" * 60)
        print("Chat Application - Type 'quit' to exit, 'clear' to reset")
        print("=" * 60 + "\n")

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'clear':
                    self.clear_history()
                    print("\nConversation history cleared.\n")
                    continue

                if not user_input:
                    continue

                # Generate response
                self.chat(user_input, stream=True)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python 01_simple_chat_app.py <model_path>")
        print("\nExample:")
        print("  python 01_simple_chat_app.py ./models/llama-2-7b-chat.Q4_K_M.gguf")
        sys.exit(1)

    model_path = sys.argv[1]

    # Create and run chat app
    app = SimpleChatApp(
        model_path=model_path,
        system_prompt="You are a helpful, friendly AI assistant.",
        n_ctx=2048,
        n_gpu_layers=35  # Adjust based on your GPU
    )

    app.run()


if __name__ == "__main__":
    main()
