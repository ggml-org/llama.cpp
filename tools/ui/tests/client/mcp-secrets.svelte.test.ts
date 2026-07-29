import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { McpSecretsService } from '$lib/services/mcp-secrets.service';
import { EncryptionService } from '$lib/services/encryption.service';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { config, settingsStore } from '$lib/stores/settings.svelte';
import {
	ENCRYPTION_META_LOCALSTORAGE_KEY,
	MCP_SECRETS_LOCALSTORAGE_KEY,
	SETTINGS_KEYS
} from '$lib/constants';

const HEADERS = '{"Authorization":"Bearer token-123"}';

let savedServers: unknown;

beforeEach(() => {
	savedServers = config().mcpServers;
});

afterEach(async () => {
	settingsStore.updateConfig(SETTINGS_KEYS.MCP_SERVERS, savedServers as string);
	EncryptionService.disable();
	localStorage.removeItem(ENCRYPTION_META_LOCALSTORAGE_KEY);
	localStorage.removeItem(MCP_SECRETS_LOCALSTORAGE_KEY);
	await McpSecretsService.load();
});

describe('McpSecretsService', () => {
	it('stores headers as plaintext JSON when encryption is disabled', async () => {
		await McpSecretsService.setHeaders('srv-1', HEADERS);

		const raw = localStorage.getItem(MCP_SECRETS_LOCALSTORAGE_KEY);
		expect(raw).not.toBeNull();
		expect(EncryptionService.isEncryptedValue(raw!)).toBe(false);
		expect(JSON.parse(raw!)).toEqual({ 'srv-1': HEADERS });

		await McpSecretsService.load();
		expect(McpSecretsService.getHeaders('srv-1')).toBe(HEADERS);
	});

	it('stores headers encrypted when encryption is enabled', async () => {
		await EncryptionService.setupWithPassphrase('pw');

		await McpSecretsService.setHeaders('srv-1', HEADERS);

		const raw = localStorage.getItem(MCP_SECRETS_LOCALSTORAGE_KEY);
		expect(EncryptionService.isEncryptedValue(raw!)).toBe(true);

		await McpSecretsService.load();
		expect(McpSecretsService.getHeaders('srv-1')).toBe(HEADERS);
	});

	it('keeps the cache empty while locked and rejects writes', async () => {
		await EncryptionService.setupWithPassphrase('pw');
		await McpSecretsService.setHeaders('srv-1', HEADERS);

		EncryptionService.lock();
		await McpSecretsService.load();
		expect(McpSecretsService.getHeaders('srv-1')).toBeUndefined();
		await expect(McpSecretsService.setHeaders('srv-2', '{}')).rejects.toThrow(
			'Encryption is locked'
		);

		expect(await EncryptionService.unlockWithPassphrase('pw')).toBe(true);
		await McpSecretsService.load();
		expect(McpSecretsService.getHeaders('srv-1')).toBe(HEADERS);
	});

	it('re-encrypts the stored map on enable and decrypts it on disable', async () => {
		await McpSecretsService.setHeaders('srv-1', HEADERS);

		await EncryptionService.setupWithPassphrase('pw');
		await McpSecretsService.persist();
		let raw = localStorage.getItem(MCP_SECRETS_LOCALSTORAGE_KEY);
		expect(EncryptionService.isEncryptedValue(raw!)).toBe(true);

		await McpSecretsService.persist({ plaintext: true });
		raw = localStorage.getItem(MCP_SECRETS_LOCALSTORAGE_KEY);
		expect(EncryptionService.isEncryptedValue(raw!)).toBe(false);
		expect(JSON.parse(raw!)).toEqual({ 'srv-1': HEADERS });
	});

	it('removes the storage key when the last secret is cleared', async () => {
		await McpSecretsService.setHeaders('srv-1', HEADERS);
		expect(localStorage.getItem(MCP_SECRETS_LOCALSTORAGE_KEY)).not.toBeNull();

		await McpSecretsService.setHeaders('srv-1', undefined);
		expect(localStorage.getItem(MCP_SECRETS_LOCALSTORAGE_KEY)).toBeNull();
	});
});

describe('mcpStore secrets integration', () => {
	function writeServers(servers: unknown[]) {
		settingsStore.updateConfig(SETTINGS_KEYS.MCP_SERVERS, JSON.stringify(servers));
	}

	it('moves inline headers into the secrets store on loadSecrets', async () => {
		writeServers([
			{ id: 'srv-1', url: 'https://example.com/mcp', enabled: true, headers: HEADERS }
		]);

		await mcpStore.loadSecrets();

		const stored = JSON.parse(config().mcpServers as string) as Record<string, unknown>[];
		expect(stored[0].headers).toBeUndefined();
		expect(McpSecretsService.getHeaders('srv-1')).toBe(HEADERS);

		const [entry] = mcpStore.getServers();
		expect(entry.headers).toBe(HEADERS);
	});

	it('does not write headers back into the config on unrelated updates', async () => {
		writeServers([
			{ id: 'srv-1', url: 'https://example.com/mcp', enabled: true, headers: HEADERS }
		]);
		await mcpStore.loadSecrets();

		mcpStore.updateServer('srv-1', { displayName: 'My server' });

		const stored = JSON.parse(config().mcpServers as string) as Record<string, unknown>[];
		expect(stored[0].headers).toBeUndefined();
		expect(stored[0].displayName).toBe('My server');
		expect(McpSecretsService.getHeaders('srv-1')).toBe(HEADERS);
	});

	it('stores new and updated headers in the secrets store only', async () => {
		writeServers([]);
		const added = mcpStore.addServer({
			url: 'https://example.com/mcp',
			enabled: true,
			headers: HEADERS
		});

		let stored = JSON.parse(config().mcpServers as string) as Record<string, unknown>[];
		expect(stored[0].headers).toBeUndefined();
		expect(McpSecretsService.getHeaders(added.id)).toBe(HEADERS);

		mcpStore.updateServer(added.id, { headers: '' });
		expect(McpSecretsService.getHeaders(added.id)).toBeUndefined();

		mcpStore.removeServer(added.id);
		stored = JSON.parse(config().mcpServers as string) as Record<string, unknown>[];
		expect(stored).toHaveLength(0);
	});

	it('preserves a sibling server inline headers on unrelated updates', async () => {
		writeServers([
			{ id: 'srv-1', url: 'https://example.com/mcp', enabled: true, headers: HEADERS },
			{ id: 'srv-2', url: 'https://example.com/other', enabled: true }
		]);

		mcpStore.updateServer('srv-2', { displayName: 'Other' });

		const stored = JSON.parse(config().mcpServers as string) as Record<string, unknown>[];
		expect(stored[0].headers).toBeUndefined();
		expect(McpSecretsService.getHeaders('srv-1')).toBe(HEADERS);
		expect(mcpStore.getServers()[0].headers).toBe(HEADERS);
	});

	it('prefers stored secrets over stale inline headers', async () => {
		await McpSecretsService.setHeaders('srv-1', HEADERS);
		writeServers([
			{ id: 'srv-1', url: 'https://example.com/mcp', enabled: true, headers: '{"X-Stale":"1"}' }
		]);

		const [entry] = mcpStore.getServers();
		expect(entry.headers).toBe(HEADERS);
	});

	it('skips migration while encryption is locked', async () => {
		await EncryptionService.setupWithPassphrase('pw');
		EncryptionService.lock();
		writeServers([
			{ id: 'srv-1', url: 'https://example.com/mcp', enabled: true, headers: HEADERS }
		]);
		await McpSecretsService.load();

		await mcpStore.migrateInlineHeadersToSecrets();

		const stored = JSON.parse(config().mcpServers as string) as Record<string, unknown>[];
		expect(stored[0].headers).toBe(HEADERS);
	});
});
