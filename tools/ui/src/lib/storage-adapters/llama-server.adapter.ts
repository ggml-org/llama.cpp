import { StorageConflictError } from './storage-conflict-error';
import { API_KEY_LOCALSTORAGE_KEY, API_STORAGE, HEADERS } from '$lib/constants';
import { PreferencesService } from '$lib/services/preferences.service';
import type { StorageAdapter, StorageAdapterDefinition, StoredConversation } from '$lib/types';
import { ApiError, apiFetch } from '$lib/utils/api-fetch';

/**
 * LlamaServerStorageAdapter - the server's /storage routes (--ui-storage)
 *
 * The server keeps one JSON file per conversation and answers a stale write
 * with 409. Requests carry the API key kept in this browser, since the
 * settings that normally carry it are themselves loaded from the server.
 */
export class LlamaServerStorageAdapter implements StorageAdapter {
	async deleteConversation(id: string): Promise<void> {
		await apiFetch(`${API_STORAGE.CONVERSATIONS}/${encodeURIComponent(id)}`, {
			headers: this.authHeaders(),
			method: 'DELETE'
		});
	}

	async getConversation(id: string): Promise<StoredConversation | null> {
		try {
			return await apiFetch<StoredConversation>(
				`${API_STORAGE.CONVERSATIONS}/${encodeURIComponent(id)}`,
				{ headers: this.authHeaders() }
			);
		} catch (error) {
			if (error instanceof ApiError && error.status === 404) return null;

			throw error;
		}
	}

	async getSettings(): Promise<Record<string, string>> {
		return await apiFetch<Record<string, string>>(API_STORAGE.SETTINGS, {
			headers: this.authHeaders()
		});
	}

	async listConversations(): Promise<Pick<StoredConversation, 'rev' | 'conv'>[]> {
		return await apiFetch(API_STORAGE.CONVERSATIONS, { headers: this.authHeaders() });
	}

	async putConversation(doc: StoredConversation): Promise<number> {
		try {
			const { rev } = await apiFetch<{ rev: number }>(
				`${API_STORAGE.CONVERSATIONS}/${encodeURIComponent(doc.conv.id)}`,
				{ body: JSON.stringify(doc), headers: this.authHeaders(), method: 'POST' }
			);

			return rev;
		} catch (error) {
			if (error instanceof ApiError && error.status === 409) {
				throw new StorageConflictError(doc.conv.id);
			}

			throw error;
		}
	}

	async updateSettings(set: Record<string, string>, remove: string[]): Promise<void> {
		await apiFetch(API_STORAGE.SETTINGS, {
			body: JSON.stringify({ remove, set }),
			headers: this.authHeaders(),
			method: 'POST'
		});
	}

	private authHeaders(): Record<string, string> {
		let apiKey: string | null = null;

		try {
			apiKey = PreferencesService.getItem(API_KEY_LOCALSTORAGE_KEY)?.trim() || null;
		} catch {
			// storage disabled, fall back to the settings store's key
		}

		return apiKey ? { [HEADERS.AUTHORIZATION]: `${HEADERS.BEARER}${apiKey}` } : {};
	}
}

export const llamaServerStorageAdapter: StorageAdapterDefinition = {
	create: () => new LlamaServerStorageAdapter(),
	description:
		"Keep conversations and settings on the server's disk, so they survive clearing browser data and are shared by every browser and port that opens this server. Anyone who can reach the server can read them.",
	id: 'llama-server',
	isAvailable: (props) => props?.ui_storage === true,
	name: 'llama-server'
};
