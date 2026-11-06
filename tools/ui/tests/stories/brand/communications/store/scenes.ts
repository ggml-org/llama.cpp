// The app scenes the Microsoft Store screenshots show: the real Llama Chat, seeded with everyday content rather than
// the test fixtures' developer questions. The scenes are stories of their own, framed in Llama for Windows' quick
// chat by StoreScreenshot.
import { MessageRole, MessageType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { conversationsStore } from '$lib/stores';

export const SCENE_CONVERSATION_ID = 'store-scene-chat';

/** the model the chat runs: a small one from llama.app's catalog, the kind an ordinary PC runs well */
export const SCENE_MODEL_ID = 'ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M';
export const SCENE_MODEL = 'gemma-4-E4B-it-Q4_K_M.gguf';
export const SCENE_MODEL_PATH = `C:\\Users\\sam\\Llama\\models\\${SCENE_MODEL}`;

/** a scene's story id, for the screenshots' window */
export const SCENES = {
	chat: 'communications-applications-store-scenes--chat'
} as const;

export type Scene = keyof typeof SCENES;

const minutesAgo = (minutes: number) => Date.now() - minutes * 60_000;
const turn = (
	id: string,
	parent: string | null,
	children: string[],
	fields: Partial<DatabaseMessage>
): DatabaseMessage => ({
	children,
	content: '',
	convId: SCENE_CONVERSATION_ID,
	id,
	parent,
	role: MessageRole.USER,
	timestamp: minutesAgo(6),
	toolCalls: '',
	type: MessageType.TEXT,
	...fields
});

/** an everyday conversation: a note to write, in plain words */
export const sceneConversation: ExportedConversation = {
	conv: {
		currNode: 'a1',
		id: SCENE_CONVERSATION_ID,
		lastModified: minutesAgo(1),
		name: 'A note to the landlord'
	},
	messages: [
		turn('root', null, ['u1'], { role: MessageRole.SYSTEM, type: MessageType.ROOT }),
		turn('u1', 'root', ['a1'], {
			content: 'My heater is broken. Can you help me write a polite note to my landlord?',
			timestamp: minutesAgo(2)
		}),
		turn('a1', 'u1', [], {
			content:
				'Of course:\n\n*Hi Anna, the heater in my flat has not worked since Monday. Could you arrange a repair this week? Thank you, Sam*',
			model: SCENE_MODEL_ID,
			role: MessageRole.ASSISTANT,
			timestamp: minutesAgo(1),
			timings: { predicted_ms: 2900, predicted_n: 41, prompt_ms: 140, prompt_n: 46 }
		})
	]
};

/** writes the conversation to IndexedDB and opens it, as the chat route does */
export async function openSceneConversation(): Promise<void> {
	await DatabaseService.deleteConversation(SCENE_CONVERSATION_ID).catch(() => {});
	await DatabaseService.importConversations([structuredClone(sceneConversation)]);
	await conversationsStore.loadConversations();
	await conversationsStore.loadConversation(SCENE_CONVERSATION_ID);
}
