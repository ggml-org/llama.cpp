/**
 * MemoryService - persistent memory for the model, stored in IndexedDB.
 *
 * Three browser tools form the whole surface of the memory protocol:
 * - memory_open: the index, or the full body of the named entries
 * - memory_write: create an entry, edit its body, refresh its description
 * - memory_drop: remove an entry
 *
 * A mutation reports the index line of the entry it touched, the model
 * holding the rest of the index from when it opened it. Edits anchor on a
 * unique string and splice the body by index, so the replacement text lands
 * verbatim whatever it contains. Byte counts are those of the markdown
 * serialization of an entry, matching a file-based memory store byte for
 * byte. Mutations run in a Dexie read-write transaction, keeping a
 * read-modify-write atomic under parallel tool calls.
 */

import {
	MEMORY_DONE,
	MEMORY_EMPTY_INDEX,
	MEMORY_ENTRY_LIMIT_BYTES_DEFAULT,
	MEMORY_NAME_PATTERN,
	MEMORY_NAME_SEPARATOR,
	NEWLINE,
	parseMemoryGroups
} from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { DatabaseMemoryEntry, ExportedMemory, ToolExecutionResult } from '$lib/types';

const BLOCK_SEPARATOR = `${NEWLINE}${NEWLINE}`;
const COLUMN_GAP = '  ';
const LIST_SEPARATOR = ', ';
const WHITESPACE_RUN = /\s+/g;
const SPACE = ' ';
const STAMP_PAD_LENGTH = 2;
const STAMP_PAD_CHAR = '0';

/**
 * Format an epoch timestamp as YYYY-MM-DD HH:MM:SS in local time.
 */
export function memoryStamp(epochMs: number): string {
	const date = new Date(epochMs);
	const pad = (value: number) => String(value).padStart(STAMP_PAD_LENGTH, STAMP_PAD_CHAR);

	return (
		`${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
		`${SPACE}${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
	);
}

const MEMORY_WRITE_KEYS = ['name', 'description', 'old_str', 'new_str'];
const textEncoder = new TextEncoder();

/**
 * UTF-8 byte count of a text, the size a filesystem stores it at.
 */
export function memoryByteLength(text: string): number {
	return textEncoder.encode(text).length;
}

/**
 * Close a body on a line break, the shape every stored body keeps so an
 * append always lands on its own line.
 */
export function closeMemoryBody(body: string): string {
	return body.length > 0 && !body.endsWith(NEWLINE) ? `${body}${NEWLINE}` : body;
}

/**
 * Serialize an entry the way a file-based store lays it out. The index byte
 * counts are those of this form, so they match such a store byte for byte.
 */
export function serializeMemoryEntry(description: string, body: string): string {
	return `---${NEWLINE}description: ${description}${NEWLINE}---${NEWLINE}${NEWLINE}${closeMemoryBody(body)}`;
}

/**
 * Fold a description onto one line, every whitespace run replaced by a
 * single space, so it occupies one row of the index.
 */
export function foldMemoryDescription(description: string): string {
	return description.replace(WHITESPACE_RUN, SPACE).trim();
}

/**
 * Validation error for a name that does not match <group>/<slug>, null when
 * the shape is valid. The shape alone gates a read and a drop, so an entry
 * whose group left the configuration stays readable and removable.
 */
export function checkMemoryNameShape(name: string): string | null {
	const [group, slug, ...rest] = name.split(MEMORY_NAME_SEPARATOR);

	if (
		rest.length > 0 ||
		!MEMORY_NAME_PATTERN.test(group ?? '') ||
		!MEMORY_NAME_PATTERN.test(slug ?? '')
	) {
		return `Invalid name "${name}", expected <group>${MEMORY_NAME_SEPARATOR}<slug> in [a-z0-9-]`;
	}

	return null;
}

/**
 * Validation error for a name that fails the shape or lands outside a
 * configured group, null when the name is valid. Writes land under a
 * configured group only.
 */
export function checkMemoryName(name: string, groups: string[]): string | null {
	const shape = checkMemoryNameShape(name);

	if (shape) return shape;

	const [group] = name.split(MEMORY_NAME_SEPARATOR);

	if (!groups.includes(group)) {
		return `Unknown group "${group}" in "${name}", groups are ${groups.join(LIST_SEPARATOR)}`;
	}

	return null;
}

function entryBytes(entry: DatabaseMemoryEntry): number {
	return memoryByteLength(serializeMemoryEntry(entry.description, entry.body));
}

function indexRow(entry: DatabaseMemoryEntry): string[] {
	return [entry.name, String(entryBytes(entry)), entry.description];
}

function columns(rows: string[][]): string {
	const nameWidth = Math.max(...rows.map((row) => row[0].length));
	const bytesWidth = Math.max(...rows.map((row) => row[1].length));

	return rows
		.map(
			(row) =>
				`${row[0].padEnd(nameWidth)}${COLUMN_GAP}${row[1].padStart(bytesWidth)}${COLUMN_GAP}${row[2]}`
		)
		.join(NEWLINE);
}

function indexLine(entry: DatabaseMemoryEntry): string {
	return columns([indexRow(entry)]);
}

function textResult(content: string, isError = false): ToolExecutionResult {
	return { content, isError };
}

export class MemoryService {
	/**
	 * Remove every entry.
	 */
	static async clearEntries(): Promise<void> {
		return DatabaseService.clearMemoryEntries();
	}

	/**
	 * Remove one entry by name.
	 *
	 * @returns False when the entry did not exist
	 */
	static async deleteEntry(name: string): Promise<boolean> {
		return DatabaseService.deleteMemoryEntry(name);
	}

	/**
	 * Execute one of the three memory tools.
	 */
	static async executeTool(
		toolName: string,
		args: Record<string, unknown>
	): Promise<ToolExecutionResult> {
		switch (toolName) {
			case BuiltInTool.BROWSER_MEMORY_OPEN:
				return MemoryService.open(args);
			case BuiltInTool.BROWSER_MEMORY_WRITE:
				return DatabaseService.memoryTransaction(() => MemoryService.write(args));
			case BuiltInTool.BROWSER_MEMORY_DROP:
				return DatabaseService.memoryTransaction(() => MemoryService.drop(args));
			default:
				return textResult(`Unknown memory tool: ${toolName}`, true);
		}
	}

	/**
	 * Snapshot of every entry, the shape a memory export file carries.
	 */
	static async exportEntries(): Promise<ExportedMemory> {
		return { entries: await MemoryService.listEntries() };
	}

	/**
	 * Merge the entries of an export file into memory. An entry whose name is
	 * already taken stays untouched. A malformed file is refused whole, so an
	 * import never writes part of one.
	 */
	static async importEntries(data: unknown): Promise<{ imported: number; skipped: number }> {
		const entries = (data as ExportedMemory | null)?.entries;

		if (!Array.isArray(entries)) {
			throw new Error('Invalid memory file: missing entries');
		}

		for (const entry of entries) {
			if (!isMemoryEntry(entry)) {
				throw new Error('Invalid memory file: malformed entry');
			}
		}

		return DatabaseService.memoryTransaction(async () => {
			let imported = 0;
			let skipped = 0;

			for (const entry of entries) {
				const existing = await DatabaseService.getMemoryEntry(entry.name);

				if (existing) {
					skipped++;

					continue;
				}

				await DatabaseService.putMemoryEntry({
					body: closeMemoryBody(entry.body),
					description: foldMemoryDescription(entry.description),
					name: entry.name,
					updated: entry.updated
				});
				imported++;
			}

			return { imported, skipped };
		});
	}

	/**
	 * Every entry, sorted by name, the listing a management UI renders.
	 */
	static async listEntries(): Promise<DatabaseMemoryEntry[]> {
		return DatabaseService.getMemoryEntries();
	}

	/**
	 * Remove an entry.
	 */
	private static async drop(args: Record<string, unknown>): Promise<ToolExecutionResult> {
		const name = String(args.name ?? '');
		const invalid = checkMemoryNameShape(name);

		if (invalid) return textResult(invalid, true);

		const removed = await DatabaseService.deleteMemoryEntry(name);

		if (!removed) {
			return textResult(`No such entry "${name}"`, true);
		}

		return textResult(`Dropped ${name}`);
	}

	private static groups(): string[] {
		return parseMemoryGroups(String(settingsStore.config.memoryGroups));
	}

	private static limit(): number {
		const value = Number(settingsStore.config.memoryEntryLimitBytes);

		return Number.isFinite(value) && value > 0 ? value : MEMORY_ENTRY_LIMIT_BYTES_DEFAULT;
	}

	/**
	 * Read the index, or the full body of the named entries.
	 */
	private static async open(args: Record<string, unknown>): Promise<ToolExecutionResult> {
		const names = Array.isArray(args.names) ? args.names.map(String) : [];

		if (names.length === 0) {
			const entries = await DatabaseService.getMemoryEntries();

			return textResult(entries.length === 0 ? MEMORY_EMPTY_INDEX : columns(entries.map(indexRow)));
		}

		const limit = MemoryService.limit();
		const blocks: string[] = [];

		let opened = 0;

		for (const name of names) {
			const invalid = checkMemoryNameShape(name);

			if (invalid) {
				blocks.push(`${name}: ${invalid}`);

				continue;
			}

			const entry = await DatabaseService.getMemoryEntry(name);

			if (!entry) {
				blocks.push(`${name}: no such entry`);

				continue;
			}

			opened++;

			const bytes = entryBytes(entry);

			blocks.push(
				`[${name}] [updated: ${memoryStamp(entry.updated)}] [size: ${bytes} of ${limit} bytes, ${limit - bytes} free]${NEWLINE}` +
					`${entry.description}${BLOCK_SEPARATOR}${entry.body.trimEnd()}`
			);
		}

		return textResult(blocks.join(BLOCK_SEPARATOR), opened === 0);
	}

	/**
	 * Persist an entry under the size cap and report the mutation followed by
	 * the index line of the entry. A write past the cap is refused, the entry
	 * keeps its previous content.
	 */
	private static async store(
		name: string,
		description: string,
		body: string,
		report: string
	): Promise<ToolExecutionResult> {
		const limit = MemoryService.limit();
		const bytes = memoryByteLength(serializeMemoryEntry(description, body));

		if (bytes > limit) {
			return textResult(
				`Entry "${name}" would reach ${bytes} bytes, over the ${limit} limit, split the subject or shorten it`,
				true
			);
		}

		const entry: DatabaseMemoryEntry = {
			body: closeMemoryBody(body),
			description,
			name,
			updated: Date.now()
		};

		await DatabaseService.putMemoryEntry(entry);

		return textResult(`${report}${NEWLINE}${indexLine(entry)}`);
	}

	/**
	 * Create an entry, edit its body, refresh its description.
	 */
	private static async write(args: Record<string, unknown>): Promise<ToolExecutionResult> {
		const name = String(args.name ?? '');
		const invalid = checkMemoryName(name, MemoryService.groups());

		if (invalid) return textResult(invalid, true);

		// An unknown key carries the text when its name is wrong, and the call
		// then removes old_str or does nothing at all
		const unknown = Object.keys(args).filter((key) => !MEMORY_WRITE_KEYS.includes(key));

		if (unknown.length > 0) {
			return textResult(
				`Unknown argument ${unknown.join(LIST_SEPARATOR)}, the text goes in old_str and new_str`,
				true
			);
		}

		const oldStr = args.old_str == null ? null : String(args.old_str);
		const newStr = args.new_str == null ? null : String(args.new_str);
		const folded =
			args.description == null ? null : foldMemoryDescription(String(args.description));
		const entry = await DatabaseService.getMemoryEntry(name);

		if (!entry) {
			if (oldStr !== null) {
				return textResult(`No such entry "${name}", nothing to replace`, true);
			}

			if (!folded) {
				return textResult(`Creating "${name}" needs a description`, true);
			}

			return MemoryService.store(name, folded, newStr === null ? '' : newStr, `Created ${name}`);
		}

		let body = entry.body;

		const done: string[] = [];

		if (oldStr !== null) {
			const at = body.indexOf(oldStr);

			if (at < 0) {
				return textResult(`String not found in ${name}, nothing written`, true);
			}

			const end = at + oldStr.length;

			if (body.indexOf(oldStr, end) >= 0) {
				return textResult(
					`String found ${body.split(oldStr).length - 1} times in ${name}, must be unique, nothing written`,
					true
				);
			}

			if (newStr === null) {
				body = body.slice(0, at) + body.slice(body[end] === NEWLINE ? end + 1 : end);
				done.push(MEMORY_DONE.REMOVED);
			} else {
				body = body.slice(0, at) + newStr + body.slice(end);
				done.push(MEMORY_DONE.REPLACED);
			}
		} else if (newStr !== null) {
			body =
				body.length === 0 || body.endsWith(NEWLINE)
					? `${body}${newStr}${NEWLINE}`
					: `${body}${NEWLINE}${newStr}${NEWLINE}`;
			done.push(MEMORY_DONE.APPENDED);
		}

		if (folded !== null && folded !== entry.description) {
			done.push(MEMORY_DONE.DESCRIPTION_REFRESHED);
		}

		if (done.length === 0) {
			return textResult(`Nothing to do on ${name}`, true);
		}

		const description = folded === null ? entry.description : folded;

		if (!description) {
			return textResult(`Entry "${name}" needs a non empty description`, true);
		}

		return MemoryService.store(name, description, body, `${name}: ${done.join(LIST_SEPARATOR)}`);
	}
}

/**
 * Structural check of one entry from an export file. The name keeps the
 * <group>/<slug> shape whatever groups are configured, the data outlives the
 * configuration.
 */
function isMemoryEntry(value: unknown): value is DatabaseMemoryEntry {
	const entry = value as DatabaseMemoryEntry | null;

	if (!entry || typeof entry !== 'object') return false;

	if (
		typeof entry.name !== 'string' ||
		typeof entry.description !== 'string' ||
		typeof entry.body !== 'string'
	) {
		return false;
	}

	if (typeof entry.updated !== 'number' || !Number.isFinite(entry.updated)) return false;

	if (foldMemoryDescription(entry.description).length === 0) return false;

	return checkMemoryNameShape(entry.name) === null;
}
