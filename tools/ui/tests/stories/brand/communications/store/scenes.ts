// The app scenes the Microsoft Store screenshots show: the real Llama Chat, seeded with everyday content rather than
// the test fixtures' developer questions: a note to write, a lease to read, a notification to understand, and a line
// to make friendlier, asked from another app through Alt + Space. The scenes are stories of their own, framed in Llama
// for Windows' quick chat by StoreScreenshot.
import lowDiskSpace from './assets/low-disk-space.png?inline';
import { AttachmentType, MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { conversationsStore } from '$lib/stores';

/** the model the chat runs: a small one from llama.app's catalog, the kind an ordinary PC runs well */
export const SCENE_MODEL_ID = 'ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M';
export const SCENE_MODEL = 'gemma-4-E4B-it-Q4_K_M.gguf';
export const SCENE_MODEL_PATH = `C:\\Users\\sam\\Llama\\models\\${SCENE_MODEL}`;

/** a scene's story id, for the screenshots' window */
export const SCENES = {
	chat: 'communications-applications-store-scenes--chat',
	document: 'communications-applications-store-scenes--document',
	quick: 'communications-applications-store-scenes--quick-chat',
	screenshot: 'communications-applications-store-scenes--screenshot'
} as const;

export type Scene = keyof typeof SCENES;

const minutesAgo = (minutes: number) => Date.now() - minutes * 60_000;

/** a question and its answer, as the app stores a conversation */
function exchange(
	id: string,
	name: string,
	question: Partial<DatabaseMessage>,
	answer: string,
	predicted: number
): ExportedConversation {
	// message ids are the database's keys across every conversation, so each scene's carry its conversation's id
	const key = (messageId: string) => `${id}-${messageId}`;
	const turn = (
		messageId: string,
		parent: string | null,
		children: string[],
		fields: Partial<DatabaseMessage>
	): DatabaseMessage => ({
		children: children.map(key),
		content: '',
		convId: id,
		id: key(messageId),
		parent: parent && key(parent),
		role: MessageRole.USER,
		timestamp: minutesAgo(2),
		toolCalls: '',
		type: MessageType.TEXT,
		...fields
	});

	return {
		conv: { currNode: key('a1'), id, lastModified: minutesAgo(1), name },
		messages: [
			turn('root', null, ['u1'], { role: MessageRole.SYSTEM, type: MessageType.ROOT }),
			turn('u1', 'root', ['a1'], question),
			turn('a1', 'u1', [], {
				content: answer,
				model: SCENE_MODEL_ID,
				role: MessageRole.ASSISTANT,
				timestamp: minutesAgo(1),
				timings: {
					predicted_ms: predicted * 70,
					predicted_n: predicted,
					prompt_ms: 140,
					prompt_n: 46
				}
			})
		]
	};
}

// the lease the document scene attaches, as the chat extracts a PDF's text; the answer flags what is in it
const lease = `RESIDENTIAL LEASE AGREEMENT

Term. This lease begins on 1 December 2026 and runs for twelve (12) months.

Rent. The tenant pays EUR 1,150 per month, due on the 1st. The deposit is two months' rent.

Renewal. This lease renews automatically for another twelve months unless either party gives written notice at least ninety (90) days before the end of the term.

Repairs. The tenant pays for any single repair under EUR 200.

Entry. The landlord may enter with 24 hours' notice.`;

/** the conversations, one per scene that has one */
export const sceneConversations: Record<Scene, ExportedConversation> = {
	chat: exchange(
		'store-scene-chat',
		'A note to the landlord',
		{ content: 'My heater is broken. Can you help me write a polite note to my landlord?' },
		'Of course:\n\n*Hi Anna, the heater in my flat has not worked since Monday. Could you arrange a repair this week? Thank you, Sam*',
		41
	),
	document: exchange(
		'store-scene-document',
		'Lease agreement',
		{
			content: 'Summarize this and flag anything unusual.',
			extra: [
				{
					base64Data: '',
					content: lease,
					name: 'lease-agreement.pdf',
					processedAsImages: false,
					size: 184_320,
					type: AttachmentType.PDF
				}
			]
		},
		"A standard 12-month lease, with two things to watch: it **renews automatically** unless you give **90 days' notice**, and **you pay for repairs under EUR 200**.",
		38
	),
	quick: exchange(
		'store-scene-quick',
		'A friendlier line',
		{ content: 'Make this sound friendlier: "Send me the report by Friday."' },
		'"Could you send me the report by Friday? Thanks so much!"',
		14
	),
	screenshot: exchange(
		'store-scene-screenshot',
		'Low disk space',
		{
			content: 'What does this mean, and how do I fix it?',
			extra: [
				{
					base64Url: lowDiskSpace,
					name: 'notification.png',
					type: AttachmentType.IMAGE
				}
			]
		},
		'Drive C:, where Windows and your apps live, is nearly full, so updates can fail and apps slow down. Open **Settings → System → Storage**: turn on **Storage Sense**, and check **Cleanup recommendations** for files you can remove.',
		64
	)
};

/** the id of a scene's conversation */
export const sceneConversationId = (scene: Scene) => sceneConversations[scene].conv.id;

/** writes a scene's conversation to IndexedDB and opens it, as the chat route does */
export async function openSceneConversation(scene: Scene): Promise<void> {
	const conversation = sceneConversations[scene];

	await DatabaseService.deleteConversation(conversation.conv.id).catch(() => {});
	await DatabaseService.importConversations([structuredClone(conversation)]);
	await conversationsStore.loadConversations();
	await conversationsStore.loadConversation(conversation.conv.id);
}
