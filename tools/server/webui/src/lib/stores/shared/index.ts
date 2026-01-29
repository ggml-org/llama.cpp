/**
 * Shared State Modules
 *
 * This directory contains dependency-free state modules that can be safely
 * imported by any store without creating circular dependencies.
 *
 * **Rules for modules in this folder:**
 * - NO imports from other stores
 * - NO imports from clients or services
 * - Only pure reactive state with no business logic
 */

export {
	activeConversationStore,
	getActiveConversationId,
	setActiveConversationId,
	isActiveConversation
} from './active-conversation.svelte';
