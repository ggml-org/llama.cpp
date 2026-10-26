/**
 * ConversationTransferService - Stateless conversation import/export layer
 *
 * Owns the session file format (one JSONL record per line: a SESSION header
 * followed by MESSAGE records), ZIP archiving and browser downloads.
 * DB access and store refreshes stay in conversationsStore.
 */

import { EXPORT_CONV, NEWLINE, ZIP_MAGIC } from '$lib/constants';
import {
	FileExtensionText,
	MimeTypeApplication,
	MimeTypeText,
	SessionRecordType
} from '$lib/enums';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';

export class ConversationTransferService {
	/**
	 * Triggers a browser download of the provided exported conversation data
	 * @param data - The exported conversation payload (a single conversation with its messages)
	 * @param filename - Filename; if omitted, a deterministic name is generated
	 */
	static downloadConversationFile(data: ExportedConversation, filename?: string): void {
		const { conv: conversation, messages: msgs } = data;

		if (!conversation) {
			console.error('Invalid data: missing conversation');

			return;
		}

		const downloadFilename =
			filename ?? ConversationTransferService.generateConversationFilename(conversation, msgs);
		const jsonl = ConversationTransferService.serializeSessionToJsonl(data);
		const blob = new Blob([jsonl], { type: MimeTypeText.JSONL });

		ConversationTransferService.triggerDownload(blob, downloadFilename);
	}

	/**
	 * Triggers a browser download of multiple conversations as a `.zip`, one
	 * `.jsonl` file per conversation.
	 * @param data - The conversations to export
	 */
	static downloadConversationsArchive(data: ExportedConversation[]): void {
		if (data.length === 0) {
			console.error('Invalid data: no conversations to export');

			return;
		}

		const usedNames = new Set<string>();
		const files: Record<string, Uint8Array> = {};

		for (const session of data) {
			const baseName = ConversationTransferService.generateConversationFilename(
				session.conv,
				session.messages
			);

			// Disambiguate any duplicate filenames within the archive.
			let entryName = baseName;
			let suffix = 1;

			while (usedNames.has(entryName)) {
				entryName = baseName.replace(
					new RegExp(`${FileExtensionText.JSONL}$`),
					`_${suffix++}${FileExtensionText.JSONL}`
				);
			}
			usedNames.add(entryName);

			files[entryName] = strToU8(ConversationTransferService.serializeSessionToJsonl(session));
		}

		const archiveName = `${new Date().toISOString().split(EXPORT_CONV.ISO_DATE_TIME_SEPARATOR)[0]}_conversations${FileExtensionText.ZIP}`;
		const zipped = zipSync(files);
		const blob = new Blob([zipped], { type: MimeTypeApplication.ZIP });

		ConversationTransferService.triggerDownload(blob, archiveName);
	}

	/**
	 * Generates a sanitized filename for a conversation export
	 * @param conversation - The conversation metadata
	 * @param msgs - Optional array of messages belonging to the conversation
	 * @returns The generated filename string
	 */
	static generateConversationFilename(
		conversation: { id?: string; name?: string },
		msgs?: DatabaseMessage[]
	): string {
		const conversationName = (conversation.name ?? '').trim().toLowerCase();
		const sanitizedName = conversationName
			.replace(EXPORT_CONV.NON_ALPHANUMERIC_REGEX, EXPORT_CONV.NONALNUM_REPLACEMENT)
			.replace(EXPORT_CONV.MULTIPLE_UNDERSCORE_REGEX, '_')
			.substring(0, EXPORT_CONV.NAME_SUFFIX_MAX_LENGTH);
		// If we have messages, use the timestamp of the newest message
		const referenceDate = msgs?.length
			? new Date(Math.max(...msgs.map((m) => m.timestamp)))
			: new Date();
		const iso = referenceDate.toISOString().slice(0, EXPORT_CONV.ISO_TIMESTAMP_SLICE);
		const formattedDate = iso
			.replace(EXPORT_CONV.ISO_DATE_TIME_SEPARATOR, EXPORT_CONV.ISO_DATE_TIME_SEPARATOR_REPLACEMENT)
			.replaceAll(EXPORT_CONV.ISO_TIME_SEPARATOR, EXPORT_CONV.ISO_TIME_SEPARATOR_REPLACEMENT);
		const trimmedConvId = conversation.id?.slice(0, EXPORT_CONV.ID_TRIM_LENGTH) ?? '';

		return `${formattedDate}_conv_${trimmedConvId}_${sanitizedName}${FileExtensionText.JSONL}`;
	}

	/**
	 * Parses an import file into conversations, accepting the current JSONL and
	 * ZIP formats as well as the legacy JSON format. The format comes from the
	 * contents, so an import works whatever the file is named.
	 * @param file - The user-selected file
	 * @returns The parsed conversations with their messages
	 */
	static async parseImportFile(file: File): Promise<ExportedConversation[]> {
		const bytes = new Uint8Array(await file.arrayBuffer());

		if (ZIP_MAGIC.every((byte, index) => bytes[index] === byte)) {
			const entries = unzipSync(bytes);
			const sessions: ExportedConversation[] = [];

			for (const [entryName, entryBytes] of Object.entries(entries)) {
				if (!entryName.toLowerCase().endsWith(FileExtensionText.JSONL)) continue;

				sessions.push(...ConversationTransferService.parseSessionsJsonl(strFromU8(entryBytes)));
			}

			return sessions;
		}

		const text = strFromU8(bytes);

		if (ConversationTransferService.isSessionsJsonl(text)) {
			return ConversationTransferService.parseSessionsJsonl(text);
		}

		// Legacy JSON format: an array of conversations or a single conversation object.
		const parsed = JSON.parse(text);

		if (Array.isArray(parsed)) {
			return parsed;
		}

		if (parsed && typeof parsed === 'object' && 'conv' in parsed && 'messages' in parsed) {
			return [parsed];
		}

		throw new Error(
			'Invalid file format: expected array of conversations or single conversation object'
		);
	}

	/**
	 * Parses the JSONL session format produced by {@link serializeSessionToJsonl}.
	 * A `SessionRecordType.SESSION` line starts a new session; following
	 * `SessionRecordType.MESSAGE` lines are appended to it. Supports multiple
	 * sessions in a single file.
	 * @param text - The JSONL file contents
	 * @returns The parsed conversations with their messages
	 */
	static parseSessionsJsonl(text: string): ExportedConversation[] {
		const sessions: ExportedConversation[] = [];

		let current: ExportedConversation | null = null;

		for (const line of text.split(NEWLINE)) {
			const trimmed = line.trim();

			if (!trimmed) continue;

			const record = JSON.parse(trimmed);

			if (record.type === SessionRecordType.SESSION) {
				// Drop the discriminator and harness marker; the rest is the conversation.
				const conv = { ...record };

				delete conv.type;
				delete conv.harness;
				current = { conv: conv as DatabaseConversation, messages: [] };
				sessions.push(current);
			} else if (record.type === SessionRecordType.MESSAGE) {
				if (!current) {
					throw new Error('Invalid JSONL: message record before any session record');
				}

				const message = record.message as DatabaseMessage;

				// `toolCalls` is parsed to an array on export; the DB stores it as a string.
				if (message.toolCalls !== undefined && typeof message.toolCalls !== 'string') {
					message.toolCalls = JSON.stringify(message.toolCalls);
				}

				current.messages.push(message);
			}
			// Ignore unknown record types for forward compatibility.
		}

		return sessions;
	}

	/**
	 * Serializes a session (a conversation with its messages) as JSONL.
	 * The first line is the session header (a `SessionRecordType.SESSION` record
	 * carrying the conversation properties); each subsequent line is a single message.
	 * @param data - The exported conversation payload
	 * @returns The JSONL string (one record per line)
	 */
	static serializeSessionToJsonl(data: ExportedConversation): string {
		const { conv, messages } = data;
		const sessionLine = JSON.stringify({
			harness: EXPORT_CONV.HARNESS,
			type: SessionRecordType.SESSION,
			...conv
		});
		const messageLines = messages.map((message: DatabaseMessage) => {
			// `toolCalls` is stored as a JSON string; drop it when empty, otherwise parse it.
			const { toolCalls, ...rest } = message;
			const normalized = toolCalls ? { ...rest, toolCalls: JSON.parse(toolCalls) } : rest;

			return JSON.stringify({ message: normalized, type: SessionRecordType.MESSAGE });
		});

		return [sessionLine, ...messageLines].join(NEWLINE);
	}

	/**
	 * Reports whether the text is the JSONL session format, whose first non-empty
	 * line is a `SessionRecordType.SESSION` record. A legacy JSON export starts
	 * with an array or an object that has no such discriminator.
	 * @param text - The file contents
	 */
	private static isSessionsJsonl(text: string): boolean {
		const trimmed = text.trimStart();
		const lineEnd = trimmed.indexOf(NEWLINE);
		const firstLine = lineEnd === -1 ? trimmed : trimmed.slice(0, lineEnd);

		try {
			return JSON.parse(firstLine).type === SessionRecordType.SESSION;
		} catch {
			// Not a standalone JSON record, so not the JSONL format.
			return false;
		}
	}

	/**
	 * Triggers a browser download of a blob under the given filename.
	 */
	private static triggerDownload(blob: Blob, filename: string): void {
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');

		a.href = url;
		a.download = filename;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}
}
