import { ChatService } from '$lib/services/chat';

export interface ChatMessageData {
	role: 'user' | 'assistant';
	content: string;
	timestamp?: Date;
}

export interface ChatState {
	messages: ChatMessageData[];
	isLoading: boolean;
	currentResponse: string;
}

export function createHandleSendMessage(
	chatService: ChatService,
	updateState: (updater: (state: ChatState) => ChatState) => void
) {
	return function handleSendMessage(message: string, currentState: ChatState) {
		if (!chatService || currentState.isLoading) return;

		const userMessage: ChatMessageData = {
			role: 'user',
			content: message,
			timestamp: new Date()
		};

		// Update state with user message and loading state
		updateState((state) => ({
			...state,
			messages: [...state.messages, userMessage],
			isLoading: true,
			currentResponse: ''
		}));

		chatService
			.sendMessage([...currentState.messages, userMessage], {
				stream: true,
				onChunk: (chunk: string) => {
					updateState((state) => {
						const newCurrentResponse = state.currentResponse + chunk;
						const lastMessage = state.messages[state.messages.length - 1];

						if (lastMessage && lastMessage.role === 'assistant') {
							// Update existing assistant message
							const updatedMessages = [...state.messages];
							updatedMessages[updatedMessages.length - 1] = {
								...lastMessage,
								content: newCurrentResponse
							};
							return {
								...state,
								messages: updatedMessages,
								currentResponse: newCurrentResponse
							};
						} else {
							// Add new assistant message
							const assistantMessage: ChatMessageData = {
								role: 'assistant',
								content: newCurrentResponse,
								timestamp: new Date()
							};
							return {
								...state,
								messages: [...state.messages, assistantMessage],
								currentResponse: newCurrentResponse
							};
						}
					});
				},
				onComplete: (response: string) => {
					updateState((state) => {
						const lastMessage = state.messages[state.messages.length - 1];
						if (lastMessage && lastMessage.role === 'assistant') {
							const updatedMessages = [...state.messages];
							updatedMessages[updatedMessages.length - 1] = {
								...lastMessage,
								content: response
							};
							return {
								...state,
								messages: updatedMessages,
								isLoading: false,
								currentResponse: ''
							};
						}
						return {
							...state,
							isLoading: false,
							currentResponse: ''
						};
					});
				},
				onError: (error: Error) => {
					console.error('Chat error:', error);
					updateState((state) => {
						const errorMessage: ChatMessageData = {
							role: 'assistant',
							content: `Error: ${error.message}`,
							timestamp: new Date()
						};
						return {
							...state,
							messages: [...state.messages, errorMessage],
							isLoading: false,
							currentResponse: ''
						};
					});
				}
			})
			.catch((error) => {
				console.error('Failed to send message:', error);
				updateState((state) => ({
					...state,
					isLoading: false,
					currentResponse: ''
				}));
			});
	};
}
