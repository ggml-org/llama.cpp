import { SvelteDate } from 'svelte/reactivity';
import { ChatService } from '../services/chat';

const chatState = $state({
	messages: [] as ChatMessageData[],
	isLoading: false,
	currentResponse: ''
});

export const chatMessages = chatState.messages;
export const isLoading = chatState.isLoading;
export const currentResponse = chatState.currentResponse;

export const chatService = new ChatService();

export async function sendMessage(message: string) {
	if (!message.trim() || chatState.isLoading) return;

	const userMessage: ChatMessageData = {
		role: 'user',
		content: message.trim(),
		timestamp: new SvelteDate()
	};

	chatState.messages.push(userMessage);
	chatState.isLoading = true;
	chatState.currentResponse = '';

	try {
		await chatService.sendMessage(chatState.messages, {
			stream: true,
			onChunk: (chunk: string) => {
				chatState.currentResponse += chunk;

				// Find or create assistant message
				const lastMessage = chatState.messages[chatState.messages.length - 1];

				if (lastMessage && lastMessage.role === 'assistant') {
					// Update existing assistant message
					lastMessage.content = chatState.currentResponse;
				} else {
					// Add new assistant message
					const assistantMessage: ChatMessageData = {
						role: 'assistant',
						content: chatState.currentResponse,
						timestamp: new SvelteDate()
					};

					chatState.messages.push(assistantMessage);
				}
			},
			onComplete: (response: string) => {
				// Ensure final message is correct
				const lastMessage = chatState.messages[chatState.messages.length - 1];

				if (lastMessage && lastMessage.role === 'assistant') {
					lastMessage.content = response;
				}
				chatState.isLoading = false;
				chatState.currentResponse = '';
			},
			onError: (error: Error) => {
				console.error('Chat error:', error);
				const errorMessage: ChatMessageData = {
					role: 'assistant',
					content: `Error: ${error.message}`,
					timestamp: new SvelteDate()
				};
				chatState.messages.push(errorMessage);
				chatState.isLoading = false;
				chatState.currentResponse = '';
			}
		});
	} catch (error) {
		console.error('Failed to send message:', error);
		chatState.isLoading = false;
		chatState.currentResponse = '';
	}
}

export function stopGeneration() {
	chatService.abort();
	chatState.isLoading = false;
	chatState.currentResponse = '';
}
