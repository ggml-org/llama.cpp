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
	import { ServerRole } from '$lib/enums';
	import { within } from 'storybook/test';

	// The app scenes the Microsoft Store screenshots frame in Llama for Windows' quick chat: the real Llama Chat with
	// everyday content, in dark mode with conversation tabs, as the Windows app shows it
	const { Story } = defineMeta({
		parameters: {
			colorScheme: 'dark',
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
		// as the Windows app opens it, with tabs, but without the per-message stats, so the question and the answer both fit
		const restoreStores = seedStores({
			activeConversation: null,
			activeMessages: [],
			config: { conversationTabs: true, showMessageStats: false },
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
</script>

<Story
	asChild
	beforeEach={chat}
	name="Chat"
	parameters={chatRoute}
	play={async ({ canvasElement }) => {
		// the chat keeps to its end; the screenshot wants the question in view, so the scene stays at the top
		await within(canvasElement).findByText(/Could you arrange a repair/);
		pinTop(canvasElement.ownerDocument.defaultView);
	}}
>
	<ChatScreen />
</Story>
