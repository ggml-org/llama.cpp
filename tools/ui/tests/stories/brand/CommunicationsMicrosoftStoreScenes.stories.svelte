<script lang="ts" module>
	import { modelServerProps, seedStores } from '../chat/form/fixtures';
	import { installFakeServer } from '../chat/screen/chat-screen.fixtures';
	import {
		openSceneConversation,
		SCENE_CONVERSATION_ID,
		SCENE_MODEL_PATH
	} from './communications/store/scenes';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreen from '$lib/components/app/chat/ChatScreen/ChatScreen.svelte';
	import ModelsManagerModelsTable from '$lib/components/app/models/ModelsManager/ModelsManagerModelsTable.svelte';
	import type {
		ModelQuantGroup,
		ModelsTableGroup
	} from '$lib/components/app/models/ModelsManager/utils';
	import { MODELS_TABLE_GROUP_LABELS } from '$lib/constants';
	import { ModelGroupKind, ModelsTableGroupKind, ServerRole } from '$lib/enums';
	import { modelsStore } from '$lib/stores/models/index.svelte';
	import type { ModelOption } from '$lib/types/models';
	import { within } from 'storybook/test';

	// The app scenes the Microsoft Store screenshots frame in a Windows window: the real Llama Chat and model manager,
	// with everyday content, always in light mode so the set reads as one
	const { Story } = defineMeta({
		parameters: {
			colorScheme: 'light',
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Applications/Store scenes'
	});

	const chatRoute = {
		sveltekit_experimental: {
			state: {
				page: {
					params: { id: SCENE_CONVERSATION_ID },
					route: { id: '/(chat)/chat/[id]' },
					url: new URL(`http://localhost/#/chat/${SCENE_CONVERSATION_ID}`)
				}
			}
		}
	};

	// a scene is a picture: once pinned, its window stays scrolled to the top, without the scroll-to-bottom button that
	// would offer to leave it, until the story changes
	let unpin = () => {};

	function pinTop(view: Window | null) {
		if (!view) return;

		const top = () => view.scrollTo(0, 0);
		const style = view.document.createElement('style');

		style.textContent = '[aria-label="Scroll to bottom"] { display: none; }';
		view.document.head.append(style);
		top();
		view.addEventListener('scroll', top);
		unpin = () => {
			view.removeEventListener('scroll', top);
			style.remove();
		};
	}

	async function chat() {
		// as a new chat, but without the per-message stats, so the question and the answer both fit the window
		const restoreStores = seedStores({
			activeConversation: null,
			activeMessages: [],
			config: { conversationTabs: false, showMessageStats: false },
			serverProps: modelServerProps({ model_path: SCENE_MODEL_PATH }),
			serverRole: ServerRole.MODEL
		});
		const restoreServer = installFakeServer();

		await openSceneConversation();

		return () => {
			unpin();
			restoreServer();
			restoreStores();
		};
	}

	const option = (model: string, name: string, contextLength?: number): ModelOption => ({
		capabilities: [],
		contextLength,
		id: model,
		model,
		name
	});
	const repo = (base: ModelOption, quants: ModelOption[]): ModelQuantGroup => ({
		base,
		key: base.model.split(':')[0],
		kind: ModelGroupKind.QUANTS,
		quants
	});
	const section = (kind: ModelsTableGroupKind, items: ModelQuantGroup[]): ModelsTableGroup => ({
		items,
		key: kind,
		kind,
		label: MODELS_TABLE_GROUP_LABELS[kind]
	});

	// small and medium models from llama.app's catalog, the kind an ordinary PC runs
	const gemma = option('ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M', 'gemma-4-E4B', 131_072);
	const omni = option('LiquidAI/LFM3-Omni-2B-GGUF:Q8_0', 'LFM3-Omni-2B', 32_768);
	const lfm = option('LiquidAI/LFM2.5-24B-A2B-GGUF:Q4_K_M', 'LFM2.5-24B-A2B', 131_072);
	const gptOss = option('ggml-org/gpt-oss-20b-GGUF:MXFP4', 'gpt-oss-20b', 131_072);

	function models() {
		// the rows read the store for favourites, hidden ids and the load status
		modelsStore.models = [gemma, omni, lfm, gptOss];
		modelsStore.favoriteModelIds = new Set([omni.model]);
		modelsStore.hiddenModelIds = new Set();
		modelsStore.routerModels = [
			{
				created: 0,
				id: gemma.model,
				object: 'model',
				owned_by: 'llamacpp',
				path: `/models/${gemma.model}`,
				status: { value: 'loaded' }
			}
		];
	}
</script>

<Story
	asChild
	beforeEach={chat}
	name="Chat"
	parameters={chatRoute}
	play={async ({ canvasElement }) => {
		// the chat keeps to its end; the screenshot wants the question in view, so the scene stays at the top
		await within(canvasElement).findByText(/Here is one you can send/);
		pinTop(canvasElement.ownerDocument.defaultView);
	}}
>
	<ChatScreen />
</Story>

<Story asChild beforeEach={models} name="Models">
	<div class="h-screen bg-background p-6">
		<ModelsManagerModelsTable
			groups={[
				section(ModelsTableGroupKind.LOADED, [repo(gemma, [gemma])]),
				section(ModelsTableGroupKind.FAVORITES, [repo(omni, [omni])]),
				section(ModelsTableGroupKind.LOCAL, [repo(lfm, [lfm]), repo(gptOss, [gptOss])])
			]}
			isFavorite={(m: ModelOption) => modelsStore.favoriteModelIds.has(m.model)}
			onSelect={() => {}}
			selectedId={gemma.id}
		/>
	</div>
</Story>
