/**
 * DocumentStoreBackend - StorageBackend over a StorageAdapter
 *
 * The adapter stores one document per conversation. Row writes are applied
 * to a cached copy of the document and recorded as ops; at the end of a
 * transaction every touched document is written back. When another client
 * wrote the document meanwhile (StorageConflictError), the fresh document is
 * fetched and the ops are replayed on it, so concurrent edits to different
 * rows both survive.
 */

import { StorageConflictError } from '$lib/storage-adapters/storage-conflict-error';
import type { StorageAdapter, StorageBackend, StoredConversation } from '$lib/types';

/** Turns a document (null: none) into its next state (null: deleted), without mutating it */
type DocumentOp = (doc: StoredConversation | null) => StoredConversation | null;

interface Batch {
	/** Cached state of each touched document before the transaction, undefined when not loaded */
	before: Map<string, StoredConversation | null | undefined>;
	ops: Map<string, DocumentOp[]>;
}

const MAX_CONFLICT_RETRIES = 3;

function groupByConversation(messages: DatabaseMessage[]): Map<string, DatabaseMessage[]> {
	const groups = new Map<string, DatabaseMessage[]>();

	for (const msg of messages) {
		const group = groups.get(msg.convId);

		if (group) group.push(msg);
		else groups.set(msg.convId, [msg]);
	}

	return groups;
}

function requireDocument(doc: StoredConversation | null, convId: string): StoredConversation {
	if (!doc) throw new Error(`Conversation ${convId} not found`);

	return doc;
}

export class DocumentStoreBackend implements StorageBackend {
	private readonly adapter: StorageAdapter;
	/** Loaded documents by conversation id, null when the store has none */
	private docs = new Map<string, StoredConversation | null>();
	/** Conversation metadata by id, null until the list is fetched */
	private index: Map<string, DatabaseConversation> | null = null;
	/** Owning conversation of every message in a loaded document */
	private messageConversation = new Map<string, string>();
	/** Writes recorded by the running transaction */
	private pending: Batch | null = null;
	private queue: Promise<unknown> = Promise.resolve();

	async addConversation(conv: DatabaseConversation): Promise<void> {
		const index = await this.getIndex();

		if (index.has(conv.id)) throw new Error(`Conversation ${conv.id} already exists`);

		const row = structuredClone(conv);

		await this.record(conv.id, (doc) => ({ conv: row, messages: doc?.messages ?? [], rev: 0 }), {
			load: false
		});
	}

	async addMessages(messages: DatabaseMessage[]): Promise<void> {
		for (const [convId, rows] of groupByConversation(structuredClone(messages))) {
			await this.record(convId, (doc) => {
				const current = requireDocument(doc, convId);

				return { ...current, messages: [...current.messages, ...rows] };
			});
		}
	}

	constructor(adapter: StorageAdapter) {
		this.adapter = adapter;
	}

	async deleteConversations(ids: string[]): Promise<void> {
		for (const id of ids) {
			await this.record(id, () => null, { load: false });
		}
	}

	async deleteMessages(ids: string[]): Promise<void> {
		const byConversation = new Map<string, Set<string>>();

		for (const id of ids) {
			const convId = this.messageConversation.get(id);

			if (!convId) continue;

			const set = byConversation.get(convId) ?? new Set<string>();

			set.add(id);
			byConversation.set(convId, set);
		}

		for (const [convId, set] of byConversation) {
			await this.record(
				convId,
				(doc) => doc && { ...doc, messages: doc.messages.filter((m) => !set.has(m.id)) }
			);
		}
	}

	async getConversations(ids: string[]): Promise<(DatabaseConversation | undefined)[]> {
		const index = await this.getIndex();

		return ids.map((id) => {
			const conv = index.get(id);

			return conv ? structuredClone(conv) : undefined;
		});
	}

	async getMessage(id: string): Promise<DatabaseMessage | undefined> {
		const convId = this.messageConversation.get(id);
		const msg = convId ? this.docs.get(convId)?.messages.find((m) => m.id === id) : undefined;

		return msg ? structuredClone(msg) : undefined;
	}

	async getMessages(convIds: string[]): Promise<DatabaseMessage[]> {
		await Promise.all(convIds.map((id) => this.load(id)));

		return convIds.flatMap((id) => structuredClone(this.docs.get(id)?.messages ?? []));
	}

	async listConversations(): Promise<DatabaseConversation[]> {
		// outside a transaction, pick up what other clients wrote since the last read
		const index = this.pending ? await this.getIndex() : await this.refreshIndex();

		return [...index.values()]
			.sort((a, b) => b.lastModified - a.lastModified || (a.id < b.id ? 1 : -1))
			.map((conv) => structuredClone(conv));
	}

	async putConversations(convs: DatabaseConversation[]): Promise<void> {
		for (const conv of structuredClone(convs)) {
			await this.record(conv.id, (doc) =>
				doc ? { ...doc, conv } : { conv, messages: [], rev: 0 }
			);
		}
	}

	async putMessages(messages: DatabaseMessage[]): Promise<void> {
		for (const [convId, rows] of groupByConversation(structuredClone(messages))) {
			await this.record(convId, (doc) => {
				const current = requireDocument(doc, convId);
				const byId = new Map(rows.map((row) => [row.id, row]));
				const kept = current.messages.map((m) => byId.get(m.id) ?? m);
				const known = new Set(current.messages.map((m) => m.id));

				return { ...current, messages: [...kept, ...rows.filter((row) => !known.has(row.id))] };
			});
		}
	}

	async transaction<T>(fn: () => Promise<T>): Promise<T> {
		// a call made while a transaction runs joins it
		if (this.pending) return await fn();

		const run = async (): Promise<T> => {
			const batch: Batch = { before: new Map(), ops: new Map() };

			let result: T;

			this.pending = batch;

			try {
				result = await fn();
			} catch (error) {
				this.rollback(batch);

				throw error;
			} finally {
				// writes that arrive while flushing start a transaction of their own
				this.pending = null;
			}

			// a failed write keeps its local edits, the next write of that document carries them
			await this.flush(batch.ops);

			return result;
		};
		const result = this.queue.then(run, run);

		this.queue = result.catch(() => {});

		return await result;
	}

	async updateConversation(
		id: string,
		updates: Partial<Omit<DatabaseConversation, 'id'>>
	): Promise<void> {
		const index = await this.getIndex();

		if (!index.has(id)) return;

		const changes = structuredClone(updates);

		await this.record(id, (doc) => doc && { ...doc, conv: { ...doc.conv, ...changes } });
	}

	async updateMessage(id: string, updates: Partial<Omit<DatabaseMessage, 'id'>>): Promise<void> {
		const convId = this.messageConversation.get(id);

		if (!convId) return;

		const changes = structuredClone(updates);

		await this.record(
			convId,
			(doc) =>
				doc && {
					...doc,
					messages: doc.messages.map((m) => (m.id === id ? { ...m, ...changes } : m))
				}
		);
	}

	/** Store a document's next state locally, keeping the revision it is based on. */
	private applyLocal(convId: string, doc: StoredConversation | null, baseRev: number): void {
		const previous = this.docs.get(convId);

		for (const msg of previous?.messages ?? []) this.messageConversation.delete(msg.id);

		if (doc) {
			this.docs.set(convId, { ...doc, rev: baseRev });
			this.index?.set(convId, doc.conv);

			for (const msg of doc.messages) this.messageConversation.set(msg.id, convId);
		} else {
			this.docs.set(convId, null);
			this.index?.delete(convId);
		}
	}

	private async flush(batch: Map<string, DocumentOp[]>): Promise<void> {
		for (const [convId, ops] of batch) {
			for (let attempt = 0; ; attempt++) {
				const doc = this.docs.get(convId) ?? null;

				try {
					if (doc) {
						doc.rev = await this.adapter.putConversation(doc);
					} else {
						await this.adapter.deleteConversation(convId);
					}

					break;
				} catch (error) {
					if (!(error instanceof StorageConflictError) || attempt >= MAX_CONFLICT_RETRIES)
						throw error;

					const fresh = await this.adapter.getConversation(convId);

					this.applyLocal(
						convId,
						ops.reduce<StoredConversation | null>((current, op) => op(current), fresh),
						fresh?.rev ?? 0
					);
				}
			}
		}
	}

	private async getIndex(): Promise<Map<string, DatabaseConversation>> {
		return this.index ?? (await this.refreshIndex());
	}

	private async load(convId: string): Promise<void> {
		if (this.docs.has(convId)) return;

		const doc = await this.adapter.getConversation(convId);

		// a concurrent load may have landed first, keep the copy that may carry local edits
		if (this.docs.has(convId)) return;

		this.applyLocal(convId, doc, doc?.rev ?? 0);
	}

	/**
	 * Record a write to one conversation document: apply it to the cached copy
	 * now and keep it for replay until the transaction is flushed.
	 */
	private async record(
		convId: string,
		op: DocumentOp,
		options: { load: boolean } = { load: true }
	): Promise<void> {
		const batch = this.pending;

		if (!batch) return await this.transaction(() => this.record(convId, op, options));

		if (options.load) await this.load(convId);

		// the transaction this write joined may have closed while the document loaded
		if (this.pending !== batch) return await this.record(convId, op, options);

		const current = this.docs.get(convId) ?? null;
		const ops = batch.ops.get(convId) ?? [];

		if (!batch.before.has(convId)) batch.before.set(convId, this.docs.get(convId));

		this.applyLocal(convId, op(current), current?.rev ?? 0);
		ops.push(op);
		batch.ops.set(convId, ops);
	}

	private async refreshIndex(): Promise<Map<string, DatabaseConversation>> {
		const entries = await this.adapter.listConversations();
		const index = new Map<string, DatabaseConversation>();

		// cached documents stay as they are; a stale one is refreshed by the 409 on its next write
		for (const { conv } of entries) {
			index.set(conv.id, conv);
		}

		this.index = index;

		return index;
	}

	/** Restore the documents a failed transaction touched, the index reloads on the next read. */
	private rollback(batch: Batch): void {
		for (const [convId, before] of batch.before) {
			if (before === undefined) {
				for (const msg of this.docs.get(convId)?.messages ?? []) {
					this.messageConversation.delete(msg.id);
				}

				this.docs.delete(convId);
			} else {
				this.applyLocal(convId, before, before?.rev ?? 0);
			}
		}

		this.index = null;
	}
}
