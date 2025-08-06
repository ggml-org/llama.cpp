/**
 * Token estimation utilities for context length validation
 */

import type { DatabaseMessage, DatabaseMessageExtra } from '$lib/types/database';
import type { ApiChatMessageData } from '$lib/types/api';

/**
 * Rough token estimation based on character count
 * Uses a conservative estimate of ~4 characters per token for most languages
 * This is a heuristic and may not be perfectly accurate, but provides a reasonable guardrail
 */
export function estimateTokenCount(text: string): number {
	if (!text) return 0;
	
	// Remove extra whitespace and normalize
	const normalizedText = text.trim().replace(/\s+/g, ' ');
	
	// Conservative estimate: ~4 characters per token
	// This accounts for various languages and encoding differences
	return Math.ceil(normalizedText.length / 4);
}

/**
 * Estimate token count for a single message including its extras (attachments)
 * @param message - The message to estimate tokens for
 * @returns The estimated token count for the message
 */
export function estimateMessageTokens(message: DatabaseMessage): number {
	let totalTokens = 0;
	
	// Count tokens in main content
	totalTokens += estimateTokenCount(message.content);
	
	// Count tokens in message extras (attachments)
	if (message.extra) {
		for (const extra of message.extra) {
			switch (extra.type) {
				case 'textFile':
					// Text files contribute their full content to token count
					totalTokens += estimateTokenCount(extra.content);
					// Add small overhead for file name formatting
					totalTokens += estimateTokenCount(`--- File: ${extra.name} ---`);
					break;
				case 'pdfFile':
					// PDF content contributes to token count
					totalTokens += estimateTokenCount(extra.content);
					totalTokens += estimateTokenCount(`--- PDF File: ${extra.name} ---`);
					break;
				case 'imageFile':
					// Images have a fixed token cost (varies by model, but ~85-170 tokens is common)
					// Using conservative estimate of 200 tokens per image
					totalTokens += 200;
					break;
				default:
					// Unknown attachment types get a small token overhead
					totalTokens += 10;
					break;
			}
		}
	}
	
	return totalTokens;
}

/**
 * Estimate total token count for a conversation including all messages
 * @param messages - The messages to estimate tokens for
 * @returns The estimated token count for the conversation
 */
export function estimateConversationTokens(messages: DatabaseMessage[]): number {
	let totalTokens = 0;
	
	for (const message of messages) {
		totalTokens += estimateMessageTokens(message);
		
		// Add small overhead for role formatting and message structure
		totalTokens += 10;
	}
	
	// Add overhead for chat template and system formatting
	totalTokens += 50;
	
	return totalTokens;
}

/**
 * Estimate tokens for a new message with extras before sending
 * @param content - The content of the new message
 * @param extras - The extras (attachments) of the new message
 * @returns The estimated token count for the new message
 */
export function estimateNewMessageTokens(content: string, extras?: DatabaseMessageExtra[]): number {
	let totalTokens = estimateTokenCount(content);
	
	if (extras) {
		for (const extra of extras) {
			switch (extra.type) {
				case 'textFile':
					totalTokens += estimateTokenCount(extra.content);
					totalTokens += estimateTokenCount(`--- File: ${extra.name} ---`);
					break;
				case 'pdfFile':
					totalTokens += estimateTokenCount(extra.content);
					totalTokens += estimateTokenCount(`--- PDF File: ${extra.name} ---`);
					break;
				case 'imageFile':
					totalTokens += 200; // Conservative estimate for image tokens
					break;
				default:
					totalTokens += 10;
					break;
			}
		}
	}
	
	// Add overhead for message formatting
	totalTokens += 10;
	
	return totalTokens;
}

/**
 * Check if adding a new message would exceed the context length
 * @param existingMessages - The existing messages in the conversation
 * @param newMessageContent - The content of the new message
 * @param newMessageExtras - The extras (attachments) of the new message
 * @param maxContextLength - The maximum allowed context length
 * @param reserveTokens - Number of tokens to reserve for response generation (default: 512)
 * @returns An object containing the estimated tokens, whether it would exceed the context length, and the maximum allowed tokens
 */
export function wouldExceedContextLength(
	existingMessages: DatabaseMessage[],
	newMessageContent: string,
	newMessageExtras: DatabaseMessageExtra[] | undefined,
	maxContextLength: number,
	reserveTokens: number = 512 // Reserve tokens for response generation
): { wouldExceed: boolean; estimatedTokens: number; maxAllowed: number } {
	const existingTokens = estimateConversationTokens(existingMessages);
	const newMessageTokens = estimateNewMessageTokens(newMessageContent, newMessageExtras);
	const totalEstimatedTokens = existingTokens + newMessageTokens;
	const maxAllowedTokens = maxContextLength - reserveTokens;
	
	return {
		wouldExceed: totalEstimatedTokens > maxAllowedTokens,
		estimatedTokens: totalEstimatedTokens,
		maxAllowed: maxAllowedTokens
	};
}
