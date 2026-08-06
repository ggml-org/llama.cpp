<script lang="ts">
	import { File, Folder } from '@lucide/svelte';
	import { getMentionBadgeLabel } from '$lib/utils';
	import { settingsStore } from '$lib/stores/settings.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import {
		MENTION_BADGE_CLASSNAME,
		MENTION_BADGE_ICON_CLASSNAME,
		PATH_SEPARATOR,
		SETTINGS_KEYS
	} from '$lib/constants';

	interface Props {
		class?: string;
		href?: string;
		name: string;
		path: string;
	}

	let { class: className = '', href, name, path }: Props = $props();

	// directories are encoded with a trailing `/` in the file:// target
	const Icon = $derived(path.endsWith(PATH_SEPARATOR) ? Folder : File);

	// Resolve the server home so ~ can abbreviate full-path labels.
	const home = $derived(toolsStore.serverHome);
	$effect(() => {
		if (typeof window === 'undefined') return;
		void toolsStore.resolveServerHome();
	});

	const label = $derived(
		getMentionBadgeLabel(
			name,
			path,
			settingsStore.getConfig(SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS),
			home
		)
	);
</script>

{#if href}
	<a
		{href}
		target="_blank"
		rel="noopener noreferrer"
		data-href={href}
		title={path}
		class={['mention-badge-link', MENTION_BADGE_CLASSNAME, className]}
	>
		<Icon class={MENTION_BADGE_ICON_CLASSNAME} aria-hidden="true" />
		<span class="shrink-0 truncate">{label}</span>
	</a>
{:else}
	<span
		data-mention-badge="true"
		data-mention-name={name}
		data-mention-path={path}
		title={path}
		class={['chat-form-mention-badge', MENTION_BADGE_CLASSNAME, className]}
	>
		<Icon class={MENTION_BADGE_ICON_CLASSNAME} aria-hidden="true" />
		<span class="shrink-0 truncate">{label}</span>
	</span>
{/if}
