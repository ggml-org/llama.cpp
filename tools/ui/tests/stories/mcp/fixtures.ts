import { HealthCheckStatus, MCPConnectionPhase, MCPLogLevel, MCPTransportType } from '$lib/enums';
import { mcpStore } from '$lib/stores';
import type {
	HealthCheckState,
	MCPCapabilitiesInfo,
	MCPConnectionLog,
	MCPServerSettingsEntry
} from '$lib/types';

const startedAt = new Date('2026-10-29T09:00:00Z');
const log = (
	offsetMs: number,
	phase: MCPConnectionPhase,
	message: string,
	level = MCPLogLevel.INFO
): MCPConnectionLog => ({
	level,
	message,
	phase,
	timestamp: new Date(startedAt.getTime() + offsetMs)
});

export const connectionLogs: MCPConnectionLog[] = [
	log(0, MCPConnectionPhase.TRANSPORT_CREATING, 'Creating streamable HTTP transport'),
	log(42, MCPConnectionPhase.TRANSPORT_READY, 'Transport ready'),
	log(118, MCPConnectionPhase.INITIALIZING, 'Sending initialize request'),
	log(204, MCPConnectionPhase.CAPABILITIES_EXCHANGED, 'Negotiated protocol 2025-06-18'),
	log(266, MCPConnectionPhase.LISTING_TOOLS, 'Listing tools'),
	log(301, MCPConnectionPhase.CONNECTED, 'Connected with 3 tools')
];

export const errorLogs: MCPConnectionLog[] = [
	log(0, MCPConnectionPhase.TRANSPORT_CREATING, 'Creating streamable HTTP transport'),
	log(5_000, MCPConnectionPhase.ERROR, 'Request timed out after 5000 ms', MCPLogLevel.ERROR)
];

export const capabilities: MCPCapabilitiesInfo = {
	client: { elicitation: { form: true }, roots: { listChanged: true }, sampling: true },
	server: {
		completions: true,
		logging: true,
		prompts: { listChanged: true },
		resources: { listChanged: true, subscribe: true },
		tools: { listChanged: true }
	}
};

const server = (id: string, name: string, enabled = true): MCPServerSettingsEntry => ({
	displayName: name,
	enabled,
	id,
	url: `https://${id}.example.com/mcp`
});

export const servers = {
	connected: server('docs', 'Docs search'),
	connecting: server('files', 'Filesystem'),
	disabled: server('weather', 'Weather', false),
	error: server('tracker', 'Issue tracker')
};

const states: Record<string, HealthCheckState> = {
	[servers.connected.id]: {
		capabilities,
		connectionTimeMs: 301,
		logs: connectionLogs,
		protocolVersion: '2025-06-18',
		serverInfo: { name: 'docs-search', version: '1.4.0' },
		status: HealthCheckStatus.SUCCESS,
		tools: [
			{ description: 'Full text search over the docs', name: 'search_docs' },
			{ description: 'Fetch one page by path', name: 'get_page' },
			{ description: 'List the table of contents', name: 'list_sections' }
		],
		transportType: MCPTransportType.STREAMABLE_HTTP
	},
	[servers.connecting.id]: {
		logs: connectionLogs.slice(0, 3),
		phase: MCPConnectionPhase.INITIALIZING,
		status: HealthCheckStatus.CONNECTING
	},
	[servers.error.id]: {
		logs: errorLogs,
		message: 'Request timed out after 5000 ms',
		phase: MCPConnectionPhase.ERROR,
		status: HealthCheckStatus.ERROR
	}
};

// health states normally come from a live probe; the setter is private, so reach it directly
type HealthAccess = {
	health: { clear: (id: string) => void; setState: (id: string, state: HealthCheckState) => void };
};

/** Seeds one health state per fixture server; returns the cleanup. */
export function seedServerHealth(): () => void {
	const { health } = mcpStore as unknown as HealthAccess;

	for (const [id, state] of Object.entries(states)) health.setState(id, state);

	return () => {
		for (const id of Object.keys(states)) health.clear(id);
	};
}
