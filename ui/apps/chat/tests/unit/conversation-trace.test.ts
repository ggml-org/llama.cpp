import { ConversationTransferService } from '#lib/services/conversation-transfer.service.ts';
import { buildConversationTrace } from '#lib/utils/index.ts';
import { describe, expect, it } from 'vitest';

function message(
	id: string,
	parent: string | null,
	children: string[],
	overrides: Partial<DatabaseMessage> = {}
): DatabaseMessage {
	return {
		children,
		content: id,
		convId: 'conv',
		id,
		parent,
		role: 'user',
		timestamp: Number(id.replace(/\D/g, '') || 0),
		type: 'text',
		...overrides
	} as DatabaseMessage;
}

// root -> m1 -> m2 -> m3
//                  \-> m4 (another branch, current)
const exported: ExportedConversation = {
	conv: {
		currNode: 'm4',
		cwd: '/home/me/project',
		groupId: 'g1',
		id: 'conv',
		lastModified: 4,
		name: 'Trace test'
	},
	group: { id: 'g1', name: 'Work' },
	messages: [
		message('root', null, ['m1'], { type: 'root' }),
		message('m1', 'root', ['m2'], {
			extra: [{ content: 'notes', name: 'a.txt', type: 'TEXT' } as unknown as DatabaseMessageExtra]
		}),
		message('m2', 'm1', ['m3', 'm4'], { role: 'assistant', toolCwd: '/home/me/project' }),
		message('m3', 'm2', []),
		message('m4', 'm2', [])
	]
};

describe('buildConversationTrace', () => {
	it('keeps the current branch only, without the root', () => {
		const trace = buildConversationTrace(exported);

		expect(trace.messages.map((m: DatabaseMessage) => m.id)).toEqual(['m1', 'm2', 'm4']);
		expect(trace.messages[1].children).toEqual(['m4']);
		expect(trace.conv.currNode).toBe('m4');
	});

	it('ends the trace at the given message', () => {
		const trace = buildConversationTrace(exported, { upToMessageId: 'm2' });

		expect(trace.messages.map((m: DatabaseMessage) => m.id)).toEqual(['m1', 'm2']);
		expect(trace.messages[1].children).toEqual([]);
		expect(trace.conv.currNode).toBe('m2');
	});

	it('leaves local paths, the group and attachments on this device', () => {
		const trace = buildConversationTrace(exported);

		expect(trace.conv).not.toHaveProperty('cwd');
		expect(trace.conv).not.toHaveProperty('groupId');
		expect(trace).not.toHaveProperty('group');
		expect(trace.messages[0]).not.toHaveProperty('extra');
		expect(trace.messages[1]).not.toHaveProperty('toolCwd');
	});

	it('keeps attachments when asked to', () => {
		const trace = buildConversationTrace(exported, { includeAttachments: true });

		expect(trace.messages[0].extra).toHaveLength(1);
	});

	it('serializes to the Session Traces Format the Hub reads', () => {
		const jsonl = ConversationTransferService.serializeSessionToJsonl(
			buildConversationTrace(exported)
		);
		const [header, ...lines] = jsonl.split('\n').map((line) => JSON.parse(line));

		expect(header).toMatchObject({ harness: 'llama.app', id: 'conv', type: 'session' });
		expect(lines).toHaveLength(3);
		expect(lines[0]).toMatchObject({ message: { content: 'm1', role: 'user' }, type: 'message' });
	});
});
