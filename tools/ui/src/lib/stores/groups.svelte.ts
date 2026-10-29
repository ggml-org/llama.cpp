/**
 * groupsStore - Named sidebar sections the user sorts chats into
 *
 * Mirrors the row-per-group table through DatabaseService, like
 * conversationsStore: every mutation awaits the row write, then applies the
 * change to the loaded list.
 */

import { browser } from '$app/environment';
import { GROUP_NAME_MAX_LENGTH } from '$lib/constants';
import { DatabaseService } from '$lib/services/database.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import type { Group } from '$lib/types';
import { uuid } from '$lib/utils';

function normalizeName(name: string): string {
	return name.trim().slice(0, GROUP_NAME_MAX_LENGTH);
}

class GroupsStore {
	/** All groups, newest first. */
	groups = $state<Group[]>([]);

	/** Whether the list was read from the database at least once. */
	isInitialized = $state(false);

	private initPromise: Promise<void> | null = null;

	/**
	 * Creates a group and puts it at the top of the list.
	 *
	 * @param name - The group's name
	 * @returns The created group, or undefined when the name is empty
	 */
	async create(name: string): Promise<Group | undefined> {
		const trimmed = normalizeName(name);

		if (!trimmed) return undefined;

		const now = Date.now();
		const group: Group = { createdAt: now, id: uuid(), name: trimmed, updatedAt: now };

		try {
			await DatabaseService.createGroup(group);
		} catch (error) {
			console.error('Failed to create group:', error);

			return undefined;
		}

		this.groups = [group, ...this.groups];

		return group;
	}

	get(id: string): Group | undefined {
		return this.groups.find((group) => group.id === id);
	}

	/**
	 * Reads every group from the database.
	 * Safe to call multiple times: concurrent callers share a single run,
	 * and a failed run can be retried by calling again.
	 */
	initialize(): Promise<void> {
		if (!browser) return Promise.resolve();

		if (this.initPromise) return this.initPromise;

		this.initPromise = (async () => {
			try {
				this.groups = await DatabaseService.getAllGroups();
				this.isInitialized = true;
			} catch (error) {
				console.error('Failed to load groups:', error);
				this.initPromise = null;
			}
		})();

		return this.initPromise;
	}

	/**
	 * Deletes a group. Its chats stay, out of any group.
	 *
	 * @param id - Group ID
	 */
	async remove(id: string): Promise<void> {
		try {
			await DatabaseService.deleteGroup(id);
		} catch (error) {
			console.error('Failed to delete group:', error);

			return;
		}

		this.groups = this.groups.filter((group) => group.id !== id);

		for (const conversation of conversationsStore.conversations) {
			if (conversation.groupId === id) {
				conversationsStore.applyConversationUpdate(conversation.id, { groupId: undefined });
			}
		}
	}

	/**
	 * Renames a group. An empty name keeps the current one.
	 *
	 * @param id - Group ID
	 * @param name - The new name
	 */
	async rename(id: string, name: string): Promise<void> {
		const trimmed = normalizeName(name);

		if (!trimmed || !this.get(id)) return;

		const changes = { name: trimmed, updatedAt: Date.now() };

		try {
			await DatabaseService.updateGroup(id, changes);
		} catch (error) {
			console.error('Failed to rename group:', error);

			return;
		}

		this.groups = this.groups.map((group) => (group.id === id ? { ...group, ...changes } : group));
	}
}

export const groupsStore = new GroupsStore();
