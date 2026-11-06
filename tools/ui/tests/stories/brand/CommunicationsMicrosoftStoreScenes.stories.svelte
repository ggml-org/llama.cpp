<script lang="ts" module>
	import { modelServerProps, seedStores } from '../chat/form/fixtures';
	import { installFakeServer } from '../chat/screen/chat-screen.fixtures';
	import {
		openSceneConversation,
		type Scene,
		SCENE_MODEL_PATH,
		sceneConversationId
	} from './communications/store/scenes';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreen from '$lib/components/app/chat/ChatScreen/ChatScreen.svelte';
	import { ServerRole } from '$lib/enums';
	import { within } from 'storybook/test';

	// The app scenes the Microsoft Store screenshots frame in Llama for Windows' quick chat: the real Llama Chat with
	// everyday content, in dark mode with conversation tabs, as the Windows app shows it: a note, a lease, a
	// notification, and a line made friendlier from another app, through Alt + Space
	const { Story } = defineMeta({
		parameters: {
			colorScheme: 'dark',
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Applications/Store scenes'
	});

	const route = (id?: string) => ({
		sveltekit_experimental: {
			state: {
				page: id
					? {
							params: { id },
							route: { id: '/(chat)/chat/[id]' },
							url: new URL(`http://localhost/#/chat/${id}`)
						}
					: { params: {}, route: { id: '/(chat)' }, url: new URL('http://localhost/#/') }
			}
		}
	});

	// a scene is a picture: once pinned, its window stays scrolled with the first message at the top (past the chat's
	// own top padding), without the scroll-to-bottom button that would offer to leave it, until the story changes
	let unpin = () => {};

	function pinTop(view: Window | null) {
		if (!view) return;

		const first = view.document.querySelector('.chat-message');
		const offset = first ? Math.max(0, first.getBoundingClientRect().top + view.scrollY - 8) : 0;
		const top = () => view.scrollTo(0, offset);
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

	/** the stores and server as the Windows app has them, with a scene's conversation open or none */
	function scene(conversation?: Scene) {
		return async () => {
			// with tabs, but without the per-message stats, so the question and the answer both fit
			const restoreStores = seedStores({
				activeConversation: null,
				activeMessages: [],
				config: { conversationTabs: true, showMessageStats: false },
				serverProps: modelServerProps({ model_path: SCENE_MODEL_PATH }),
				serverRole: ServerRole.MODEL
			});
			const restoreServer = installFakeServer();

			if (conversation) await openSceneConversation(conversation);

			return () => {
				unpin();
				restoreServer();
				restoreStores();
			};
		};
	}

	/** the chat keeps to its end; the screenshot wants the question in view, so once the answer is in it stays at the top */
	const pinOnceIn =
		(text: RegExp) =>
		async ({ canvasElement }: { canvasElement: HTMLElement }) => {
			await within(canvasElement).findByText(text);
			pinTop(canvasElement.ownerDocument.defaultView);
		};
</script>

<Story
	asChild
	beforeEach={scene('chat')}
	name="Chat"
	parameters={route(sceneConversationId('chat'))}
	play={pinOnceIn(/Could you arrange a repair/)}
>
	<ChatScreen />
</Story>

<Story
	asChild
	beforeEach={scene('document')}
	name="Document"
	parameters={route(sceneConversationId('document'))}
	play={pinOnceIn(/two things to watch/)}
>
	<ChatScreen />
</Story>

<Story
	asChild
	beforeEach={scene('screenshot')}
	name="Screenshot"
	parameters={route(sceneConversationId('screenshot'))}
	play={pinOnceIn(/Storage Sense/)}
>
	<ChatScreen />
</Story>

<Story
	asChild
	beforeEach={scene('quick')}
	name="Quick chat"
	parameters={route(sceneConversationId('quick'))}
	play={pinOnceIn(/Thanks so much/)}
>
	<ChatScreen />
</Story>
