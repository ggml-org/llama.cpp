<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { InsetList, InsetListRow, InsetListSplitRow } from '@ggml-org/llama-ui/app/sheets';
	import { Switch } from '@ggml-org/llama-ui/ui/switch';
	import { Check, Folder, Globe, Server, Trash2 } from '@lucide/svelte';

	// the native-style list of a phone's sheets: rounded groups of finger-tall rows, the divider
	// starting at the label. Shown in a float panel with the sheet's group radius, as a drawer sets it
	const { Story } = defineMeta({
		component: InsetList,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/App components/InsetList'
	});

	const PANEL_CLASS =
		'matte-float flex w-80 flex-col gap-6 rounded-2xl border p-3 [--inset-group-radius:var(--sheet-group-radius)]';
</script>

<script lang="ts">
	let webSearch = $state(true);
	let server = $state(false);
	let picked = $state('Light');
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Rows that open a page" note="icon, value, chevron">
			<div class={PANEL_CLASS}>
				<InsetList caption="A caption sits under its group." title="Files">
					<InsetListRow chevron icon={Folder}>
						<span class="shrink-0">Working directory</span>

						<span class="min-w-0 flex-1 truncate text-right text-body-sm text-muted-foreground">
							~/projects/llama
						</span>
					</InsetListRow>

					<InsetListRow chevron icon={Globe}>
						<span class="flex-1">Web search</span>
					</InsetListRow>
				</InsetList>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Rows with a switch" note="the whole row is the label">
			<div class={PANEL_CLASS}>
				<InsetList>
					<InsetListRow as="label">
						<span class="flex-1">Web search</span>

						<Switch bind:checked={webSearch} size="lg" />
					</InsetListRow>

					<InsetListRow as="label">
						<span class="flex-1">Server tools</span>

						<Switch bind:checked={server} size="lg" />
					</InsetListRow>
				</InsetList>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="A choice of one" note="a check marks the pick">
			<div class={PANEL_CLASS}>
				<InsetList title="Appearance">
					{#each ['Light', 'Dark', 'System'] as option (option)}
						<InsetListRow aria-pressed={option === picked} onclick={() => (picked = option)}>
							<span class="flex-1">{option}</span>

							{#if option === picked}
								<Check class="size-4 text-(--action-primary)" />
							{/if}
						</InsetListRow>
					{/each}
				</InsetList>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="Split rows" note="the row opens a page, the switch acts on it">
			<div class={PANEL_CLASS}>
				<InsetList title="MCP servers">
					<InsetListSplitRow icon={Server} onclick={() => {}}>
						<span class="min-w-0 flex-1 truncate">Hugging Face</span>

						<span class="text-body-sm text-muted-foreground tabular-nums">4/9</span>

						{#snippet trailing()}
							<Switch aria-label="Hugging Face tools" checked size="lg" />
						{/snippet}
					</InsetListSplitRow>

					<InsetListSplitRow dimmed icon={Server} onclick={() => {}}>
						<span class="min-w-0 flex-1 truncate">Filesystem</span>

						<span class="text-body-sm text-muted-foreground tabular-nums">0/5</span>

						{#snippet trailing()}
							<Switch aria-label="Filesystem tools" disabled size="lg" />
						{/snippet}
					</InsetListSplitRow>
				</InsetList>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="A destructive row">
			<div class={PANEL_CLASS}>
				<InsetList>
					<InsetListRow bodyClass="text-destructive" icon={Trash2} iconClass="text-destructive">
						Delete all conversations
					</InsetListRow>
				</InsetList>
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
