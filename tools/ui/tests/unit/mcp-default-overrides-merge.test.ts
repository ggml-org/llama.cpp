import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { STORAGE_APP_NAME, CONFIG_LOCALSTORAGE_KEY } from '$lib/constants';

// node env unit project has no DOM, install a minimal localStorage backed by a Map
beforeAll(() => {
	const store = new Map<string, string>();
	const polyfill: Storage = {
		get length() {
			return store.size;
		},
		clear: () => store.clear(),
		getItem: (k) => (store.has(k) ? store.get(k)! : null),
		key: (i) => Array.from(store.keys())[i] ?? null,
		removeItem: (k) => {
			store.delete(k);
		},
		setItem: (k, v) => {
			store.set(k, String(v));
		}
	};
	(globalThis as unknown as { localStorage: Storage }).localStorage = polyfill;
});

/**
 * Migration `mcp-default-overrides-merge-v1` folds the values of the parallel
 * `mcpDefaultServerOverrides` config entry onto `mcpServers[i].enabled` (the
 * single source of truth for new-chat defaults). The legacy key is kept on
 * disk for downgrade compatibility.
 */
describe('mcp-default-overrides-merge-v1 migration', () => {
	const MIGRATION_STATE_KEY = `${STORAGE_APP_NAME}.migration-state`;
	const MCP_DEFAULT_OVERRIDES_KEY = `${STORAGE_APP_NAME}.mcpDefaultServerOverrides`;

	beforeEach(async () => {
		localStorage.clear();
		// Reset the migration run counter so `runAllMigrations` is guaranteed to execute.
		await import('$lib/services/migration.service').then((mod) =>
			mod.MigrationService.resetState()
		);
	});

	afterEach(() => {
		localStorage.clear();
	});

	async function runMigrations() {
		const { MigrationService } = await import('$lib/services/migration.service');
		await MigrationService.runAllMigrations();
	}

	function readConfig(): Record<string, unknown> {
		const raw = localStorage.getItem(CONFIG_LOCALSTORAGE_KEY);

		return raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
	}

	function writeConfig(config: Record<string, unknown>) {
		localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify(config));
	}

	it('applies matching overrides onto mcpServers[i].enabled and preserves the legacy key', async () => {
		writeConfig({
			mcpServers: JSON.stringify([
				{ id: 'exa', enabled: false, url: 'https://mcp.exa.ai/mcp' },
				{ id: 'hf', enabled: false, url: 'https://huggingface.co/mcp' }
			]),
			[MCP_DEFAULT_OVERRIDES_KEY]: JSON.stringify([
				{ serverId: 'exa', enabled: true },
				{ serverId: 'hf', enabled: false }
			])
		});

		await runMigrations();

		const after = readConfig();
		const servers = JSON.parse(after.mcpServers as string) as Array<{
			id: string;
			enabled: boolean;
		}>;

		expect(servers.find((s) => s.id === 'exa')?.enabled).toBe(true);
		expect(servers.find((s) => s.id === 'hf')?.enabled).toBe(false);
		expect(MCP_DEFAULT_OVERRIDES_KEY in after).toBe(true);
	});

	it('skips override ids that do not match any configured server', async () => {
		writeConfig({
			mcpServers: JSON.stringify([{ id: 'exa', enabled: false, url: 'https://mcp.exa.ai/mcp' }]),
			[MCP_DEFAULT_OVERRIDES_KEY]: JSON.stringify([
				{ serverId: 'orphan', enabled: true },
				{ serverId: 'exa', enabled: true }
			])
		});

		await runMigrations();

		const after = readConfig();
		const servers = JSON.parse(after.mcpServers as string) as Array<{
			id: string;
			enabled: boolean;
		}>;

		expect(servers).toHaveLength(1);
		expect(servers[0].enabled).toBe(true);
		expect(MCP_DEFAULT_OVERRIDES_KEY in after).toBe(true);
	});

	it('is a no-op when there are no legacy overrides', async () => {
		writeConfig({
			mcpServers: JSON.stringify([{ id: 'exa', enabled: true, url: 'https://mcp.exa.ai/mcp' }])
		});

		await runMigrations();

		const after = readConfig();
		const servers = JSON.parse(after.mcpServers as string) as Array<{
			id: string;
			enabled: boolean;
		}>;

		expect(servers[0].enabled).toBe(true);
		expect(MCP_DEFAULT_OVERRIDES_KEY in after).toBe(false);
	});

	it('does not rewrite mcpServers when override.enabled already matches', async () => {
		const originalServers = JSON.stringify([
			{ id: 'exa', enabled: true, url: 'https://mcp.exa.ai/mcp' }
		]);

		writeConfig({
			mcpServers: originalServers,
			[MCP_DEFAULT_OVERRIDES_KEY]: JSON.stringify([{ serverId: 'exa', enabled: true }])
		});

		await runMigrations();

		const after = readConfig();
		expect(after.mcpServers).toBe(originalServers);
		expect(MCP_DEFAULT_OVERRIDES_KEY in after).toBe(true);
	});

	it('records itself as completed so subsequent loads do not re-run', async () => {
		writeConfig({
			mcpServers: JSON.stringify([{ id: 'exa', enabled: false, url: 'https://mcp.exa.ai/mcp' }]),
			[MCP_DEFAULT_OVERRIDES_KEY]: JSON.stringify([{ serverId: 'exa', enabled: true }])
		});

		const { MigrationService } = await import('$lib/services/migration.service');

		await MigrationService.runAllMigrations();

		const stateRaw = localStorage.getItem(MIGRATION_STATE_KEY);
		expect(stateRaw).not.toBeNull();
		const state = JSON.parse(stateRaw!) as { completed: string[]; failed: string[] };
		expect(state.completed).toContain('mcp-default-overrides-merge-v1');
		expect(state.failed).not.toContain('mcp-default-overrides-merge-v1');
	});
});
