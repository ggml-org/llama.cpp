<script lang="ts" module>
	import { getPace } from '../../pace';
	import {
		installFakeServer,
		openScreenConversation,
		SCREEN_CONVERSATION_ID,
		seedNewChat
	} from './chat-screen.fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreen from '$lib/components/app/chat/ChatScreen/ChatScreen.svelte';
	import { chatStore, conversationsStore, serverStore } from '$lib/stores';
	import { expect, waitFor, within } from 'storybook/test';

	const newChatRoute = {
		sveltekit_experimental: {
			state: { page: { params: {}, route: { id: '/(chat)' }, url: new URL('http://localhost/#/') } }
		}
	};

	const conversationRoute = {
		sveltekit_experimental: {
			state: {
				page: {
					params: { id: SCREEN_CONVERSATION_ID },
					route: { id: '/(chat)/chat/[id]' },
					url: new URL(`http://localhost/#/chat/${SCREEN_CONVERSATION_ID}`)
				}
			}
		}
	};

	const { Story } = defineMeta({
		component: ChatScreen,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Chat/ChatScreen/ChatScreen'
	});

	// the streamed reply ends with this sentence, so it is on screen only once the stream is done
	const REPLY_END = 'at a small cost in quality.';

	/** Stores and the fake server for a story; async so the conversation is in IndexedDB before render. */
	function setup(
		options: {
			conversation?: boolean;
			server?: Parameters<typeof installFakeServer>[0];
			/** Streams this many times slower than the pace, to leave time to act mid-reply. */
			slowStream?: number;
		} = {}
	) {
		return async (context: { globals?: Record<string, unknown> }) => {
			const pace = getPace(context);
			const restoreStores = seedNewChat();
			const restoreServer = installFakeServer({
				chunkDelayMs: pace.streamChunkMs * (options.slowStream ?? 1),
				...options.server
			});

			if (options.conversation) await openScreenConversation();

			return () => {
				chatStore.stopGeneration();
				restoreServer();
				restoreStores();
			};
		};
	}

	/** Types the message key by key at the story's pace, then sends it with Enter. */
	async function send(
		context: { canvasElement: HTMLElement; globals?: Record<string, unknown> },
		text: string
	) {
		const { pause, user } = getPace(context);
		const textarea = await within(context.canvasElement).findByRole('textbox');

		await pause();
		await user.type(textarea, text);
		await pause();
		await user.keyboard('{Enter}');
	}
</script>

<Story
	args={{ showCenteredEmpty: true }}
	beforeEach={setup()}
	name="Default"
	parameters={newChatRoute}
	play={async (context) => {
		const { canvas } = context;

		await expect(await canvas.findByText('Hello there')).toBeInTheDocument();
		await expect(canvas.getByRole('textbox')).toHaveFocus();
		await expect(canvas.getByRole('button', { name: 'Send' })).toBeDisabled();
	}}
/>

<Story
	args={{ showCenteredEmpty: true }}
	beforeEach={setup()}
	name="SendFirstMessage"
	parameters={newChatRoute}
	play={async (context) => {
		const { canvas, step } = context;

		await step('type and send a message', async () => {
			await send(context, 'What does quantization do?');
		});

		await step('the message moves into the chat and the reply streams in', async () => {
			await expect(await canvas.findByText('What does quantization do?')).toBeInTheDocument();
			await expect(
				await canvas.findByText(REPLY_END, { exact: false }, { timeout: 8000 })
			).toBeInTheDocument();
		});

		await step('the form is ready for the next message', async () => {
			await waitFor(() => expect(chatStore.isLoading).toBe(false));
			await expect(canvas.getByRole('textbox')).toHaveValue('');
			await expect(conversationsStore.activeMessages.map((m) => m.role)).toEqual([
				'user',
				'assistant'
			]);
		});
	}}
/>

<Story
	args={{ showCenteredEmpty: false }}
	beforeEach={setup({ conversation: true })}
	name="Conversation"
	parameters={conversationRoute}
	play={async (context) => {
		const { canvas } = context;

		await expect(
			await canvas.findByText('What is the difference between Q4_K_M and Q8_0?')
		).toBeInTheDocument();
		await expect(canvas.getByText('How do I run one with llama-server?')).toBeInTheDocument();
		await expect(canvas.getByRole('table')).toBeInTheDocument();
		await expect(canvas.queryByText('Hello there')).not.toBeVisible();
	}}
/>

<Story
	args={{ showCenteredEmpty: false }}
	beforeEach={setup({ conversation: true })}
	name="ContinueConversation"
	parameters={conversationRoute}
	play={async (context) => {
		const { canvas } = context;

		await canvas.findByText('How do I run one with llama-server?');

		await send(context, 'Which one fits in 8 GB of VRAM?');

		await expect(await canvas.findByText('Which one fits in 8 GB of VRAM?')).toBeInTheDocument();
		await expect(
			await canvas.findByText(REPLY_END, { exact: false }, { timeout: 8000 })
		).toBeInTheDocument();

		// the follow-up joins the existing branch: 4 earlier turns plus the new pair
		await waitFor(() => expect(conversationsStore.activeMessages).toHaveLength(6));
	}}
/>

<Story
	args={{ showCenteredEmpty: true }}
	beforeEach={setup({ slowStream: 4 })}
	name="StopGeneration"
	parameters={newChatRoute}
	play={async (context) => {
		const { canvas, step, userEvent } = context;

		await send(context, 'Tell me about quantization');

		await step('stop the reply while it streams', async () => {
			const stop = await canvas.findByRole('button', { name: 'Stop' }, { timeout: 4000 });

			await canvas.findByText('Sure.', { exact: false }, { timeout: 8000 });
			await getPace(context).pause();
			await userEvent.click(stop);
		});

		await step('the partial reply stays and the form is ready again', async () => {
			await waitFor(() => expect(chatStore.isLoading).toBe(false));
			await expect(canvas.getByText('Sure.', { exact: false })).toBeInTheDocument();
			await expect(canvas.queryByText(REPLY_END, { exact: false })).not.toBeInTheDocument();
			await expect(canvas.getByRole('button', { name: 'Send' })).toBeInTheDocument();
		});
	}}
/>

<Story
	args={{ showCenteredEmpty: true }}
	beforeEach={setup({
		server: { errorMessage: 'the request exceeds the available context size', status: 500 }
	})}
	name="ServerErrorReply"
	parameters={newChatRoute}
	play={async (context) => {
		await send(context, 'Summarize this very long document');

		// the dialog is portalled to the body, outside the canvas
		const page = within(document.body);

		await expect(await page.findByRole('alertdialog', {}, { timeout: 4000 })).toBeInTheDocument();
		await expect(page.getByText('Server Error')).toBeInTheDocument();
		await expect(
			page.getByText('the request exceeds the available context size', { exact: false })
		).toBeInTheDocument();
	}}
/>

<Story
	args={{ showCenteredEmpty: true }}
	beforeEach={async (context) => {
		const restore = await setup()(context);

		serverStore.error = 'Failed to fetch server properties: connection refused';
		serverStore.status = 500;

		return () => {
			serverStore.error = null;
			serverStore.status = null;
			restore();
		};
	}}
	name="ServerUnavailable"
	parameters={newChatRoute}
	play={async (context) => {
		const { canvas } = context;

		await expect(await canvas.findByText(/connection refused/)).toBeInTheDocument();
		await expect(canvas.getByRole('textbox')).toBeDisabled();
	}}
/>
