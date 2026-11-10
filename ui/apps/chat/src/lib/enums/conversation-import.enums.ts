/**
 * Discriminator of a record line in the JSONL conversation format. A session
 * record opens a conversation and carries its properties; every following
 * message record belongs to it.
 */
export enum SessionRecordType {
	MESSAGE = 'message',
	SESSION = 'session'
}
