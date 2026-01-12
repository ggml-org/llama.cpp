/**
 * Clients Module - Business Logic Facades
 *
 * This module exports all client classes which coordinate business logic:
 * - MCPClient: MCP connection management and tool execution
 * - ChatClient: Message operations, streaming, branching
 * - AgenticClient: Multi-turn tool loop orchestration
 * - ConversationsClient: Conversation CRUD and message management
 *
 * **Architecture:**
 * - Clients coordinate between Services (stateless API) and Stores (reactive state)
 * - Clients contain business logic, orchestration, and error handling
 * - Stores only hold reactive state and delegate to Clients
 *
 * @see services/ for stateless API operations
 * @see stores/ for reactive state
 */

// MCP Client
export { MCPClient, mcpClient } from './mcp.client';

// Chat Client
export { ChatClient, chatClient } from './chat.client';
export type { ChatStreamCallbacks, ApiProcessingState, ErrorDialogState } from './chat.client';

// Agentic Client
export { AgenticClient, agenticClient } from './agentic.client';
export type {
	AgenticFlowCallbacks,
	AgenticFlowOptions,
	AgenticFlowParams,
	AgenticFlowResult
} from './agentic.client';

// Conversations Client
export { ConversationsClient, conversationsClient } from './conversations.client';
