import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { MigrationService } from '$lib/services/migration.service';
import { McpSecretsService } from '$lib/services/mcp-secrets.service';
import { EncryptionService } from '$lib/services/encryption.service';
import {
	CONFIG_LOCALSTORAGE_KEY,
	ENCRYPTION_META_LOCALSTORAGE_KEY,
	MCP_SECRETS_LOCALSTORAGE_KEY,
	SETTINGS_KEYS
} from '$lib/constants';

const HEADERS = '{"Authorization":"Bearer token-123"}';

function writeServers(servers: unknown[]) {
	const config = JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}');
	config[SETTINGS_KEYS.MCP_SERVERS] = JSON.stringify(servers);
	localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify(config));
}

function readServers(): Record<string, unknown>[] {
	const config = JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}');
	return JSON.parse(config[SETTINGS_KEYS.MCP_SERVERS] ?? '[]');
}

beforeEach(() => {
	MigrationService.resetState();
});

afterEach(async () => {
	MigrationService.resetState();
	EncryptionService.disable();
	localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
	localStorage.removeItem(ENCRYPTION_META_LOCALSTORAGE_KEY);
	localStorage.removeItem(MCP_SECRETS_LOCALSTORAGE_KEY);
	await McpSecretsService.load();
});

describe('migration phases', () => {
	it('marks content-touching migrations as post-unlock', () => {
		const migrations = MigrationService.getMigrations();

		expect(migrations.find((m) => m.id === 'legacy-message-format-v2')?.phase).toBe('post-unlock');
		expect(migrations.find((m) => m.id === 'mcp-headers-to-secrets-v1')?.phase).toBe('post-unlock');
	});

	it('does not run post-unlock migrations in the boot pass', async () => {
		writeServers([
			{ id: 'srv-1', url: 'https://example.com/mcp', enabled: true, headers: HEADERS }
		]);

		await MigrationService.runAllMigrations();

		expect(readServers()[0].headers).toBe(HEADERS);
		expect(MigrationService.isCompleted('mcp-headers-to-secrets-v1')).toBe(false);
	});
});

describe('mcp-headers-to-secrets-v1 migration', () => {
	it('moves inline headers into the secrets store and strips them from config', async () => {
		writeServers([
			{ id: 'srv-1', url: 'https://example.com/mcp', enabled: true, headers: HEADERS },
			{ id: 'srv-2', url: 'https://example.com/other', enabled: false }
		]);

		await MigrationService.runAllMigrations('post-unlock');

		const servers = readServers();
		expect(servers[0].headers).toBeUndefined();
		expect(servers[1].headers).toBeUndefined();
		expect(McpSecretsService.getHeaders('srv-1')).toBe(HEADERS);
		expect(McpSecretsService.getHeaders('srv-2')).toBeUndefined();
	});

	it('freezes the resolved id for servers without one', async () => {
		writeServers([{ url: 'https://example.com/mcp', enabled: true, headers: HEADERS }]);

		await MigrationService.runAllMigrations('post-unlock');

		const servers = readServers();
		const id = servers[0].id as string;
		expect(id).toBe('LlamaUI-MCP-Server-1');
		expect(servers[0].headers).toBeUndefined();
		expect(McpSecretsService.getHeaders(id)).toBe(HEADERS);
	});

	it('keeps existing secrets over stale inline headers', async () => {
		await McpSecretsService.setHeaders('srv-1', HEADERS);
		writeServers([
			{ id: 'srv-1', url: 'https://example.com/mcp', enabled: true, headers: '{"X-Stale":"1"}' }
		]);

		await MigrationService.runAllMigrations('post-unlock');

		expect(McpSecretsService.getHeaders('srv-1')).toBe(HEADERS);
		expect(readServers()[0].headers).toBeUndefined();
	});

	it('encrypts the moved headers when encryption is enabled', async () => {
		await EncryptionService.setupWithPassphrase('pw');
		writeServers([
			{ id: 'srv-1', url: 'https://example.com/mcp', enabled: true, headers: HEADERS }
		]);

		await MigrationService.runAllMigrations('post-unlock');

		const raw = localStorage.getItem(MCP_SECRETS_LOCALSTORAGE_KEY);
		expect(EncryptionService.isEncryptedValue(raw!)).toBe(true);
		expect(McpSecretsService.getHeaders('srv-1')).toBe(HEADERS);
		expect(readServers()[0].headers).toBeUndefined();
	});
});
