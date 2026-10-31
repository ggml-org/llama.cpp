/** Thrown by a storage adapter when a write is based on a revision the store no longer holds. */
export class StorageConflictError extends Error {
	constructor(convId: string) {
		super(`Conversation ${convId} was changed by another client`);
		this.name = 'StorageConflictError';
	}
}
