export const CLI_FLAGS = {
	AGENT: '--agent',
	API_KEY: '--api-key',
	MCP_PROXY: '--ui-mcp-proxy',
	/** Multimodal projector path; unlocks vision/audio for the model. */
	MMPROJ: '--mmproj',
	/** Draft model weights path (long form); the router records it per model. */
	MODEL_DRAFT: '--model-draft',
	/** Draft model weights path (short form). */
	MODEL_DRAFT_SHORT: '-md',
	SLOTS: '--slots',
	TOOLS: '--tools'
} as const;
