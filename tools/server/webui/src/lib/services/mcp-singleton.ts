import { browser } from '$app/environment';
import { MCPClient } from '$lib/mcp';
import { buildMcpClientConfig } from '$lib/config/mcp';
import { config } from '$lib/stores/settings.svelte';

const globalState = globalThis as typeof globalThis & {
	__llamaMcpClient?: MCPClient;
	__llamaMcpInitPromise?: Promise<MCPClient | undefined>;
	__llamaMcpConfigSignature?: string;
	__llamaMcpInitConfigSignature?: string;
};

function serializeConfigSignature(): string | undefined {
	const mcpConfig = buildMcpClientConfig(config());
	return mcpConfig ? JSON.stringify(mcpConfig) : undefined;
}

async function shutdownClient(): Promise<void> {
	if (!globalState.__llamaMcpClient) return;

	const clientToShutdown = globalState.__llamaMcpClient;
	globalState.__llamaMcpClient = undefined;
	globalState.__llamaMcpConfigSignature = undefined;

	try {
		await clientToShutdown.shutdown();
	} catch (error) {
		console.error('[MCP] Failed to shutdown client:', error);
	}
}

async function bootstrapClient(
	signature: string,
	mcpConfig: ReturnType<typeof buildMcpClientConfig>
): Promise<MCPClient | undefined> {
	if (!browser || !mcpConfig) {
		return undefined;
	}

	const client = new MCPClient(mcpConfig);
	globalState.__llamaMcpInitConfigSignature = signature;

	const initPromise = client
		.initialize()
		.then(() => {
			// Ignore initialization if config changed during bootstrap
			if (globalState.__llamaMcpInitConfigSignature !== signature) {
				void client.shutdown().catch((shutdownError) => {
					console.error(
						'[MCP] Failed to shutdown stale client after config change:',
						shutdownError
					);
				});
				return undefined;
			}

			globalState.__llamaMcpClient = client;
			globalState.__llamaMcpConfigSignature = signature;
			return client;
		})
		.catch((error) => {
			console.error('[MCP] Failed to initialize client:', error);

			// Cleanup global references on error
			if (globalState.__llamaMcpClient === client) {
				globalState.__llamaMcpClient = undefined;
			}
			if (globalState.__llamaMcpConfigSignature === signature) {
				globalState.__llamaMcpConfigSignature = undefined;
			}

			void client.shutdown().catch((shutdownError) => {
				console.error('[MCP] Failed to shutdown client after init error:', shutdownError);
			});
			return undefined;
		})
		.finally(() => {
			// Clear init promise only if it's OUR promise
			if (globalState.__llamaMcpInitPromise === initPromise) {
				globalState.__llamaMcpInitPromise = undefined;
				// Clear init signature only if it's still ours
				if (globalState.__llamaMcpInitConfigSignature === signature) {
					globalState.__llamaMcpInitConfigSignature = undefined;
				}
			}
		});

	globalState.__llamaMcpInitPromise = initPromise;
	return initPromise;
}

export function getMcpClient(): MCPClient | undefined {
	return globalState.__llamaMcpClient;
}

export async function ensureMcpClient(): Promise<MCPClient | undefined> {
	const signature = serializeConfigSignature();

	// Configuration removed: shut down active client if present
	if (!signature) {
		// Wait for any in-flight init to complete before shutdown
		if (globalState.__llamaMcpInitPromise) {
			await globalState.__llamaMcpInitPromise;
		}
		await shutdownClient();
		globalState.__llamaMcpInitPromise = undefined;
		globalState.__llamaMcpInitConfigSignature = undefined;
		return undefined;
	}

	// Client already initialized with correct config
	if (globalState.__llamaMcpClient && globalState.__llamaMcpConfigSignature === signature) {
		return globalState.__llamaMcpClient;
	}

	// Init in progress with correct config
	if (
		globalState.__llamaMcpInitPromise &&
		globalState.__llamaMcpInitConfigSignature === signature
	) {
		return globalState.__llamaMcpInitPromise;
	}

	// Config changed - wait for in-flight init before shutdown
	if (
		globalState.__llamaMcpInitPromise &&
		globalState.__llamaMcpInitConfigSignature !== signature
	) {
		await globalState.__llamaMcpInitPromise;
	}

	// Shutdown if config changed
	if (globalState.__llamaMcpConfigSignature !== signature) {
		await shutdownClient();
	}

	// Bootstrap new client
	const mcpConfig = buildMcpClientConfig(config());
	return bootstrapClient(signature, mcpConfig);
}
