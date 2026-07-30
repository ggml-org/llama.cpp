<script lang="ts">
	import { File, Folder } from '@lucide/svelte';
	import { MENTION_BADGE_CLASSNAME, MENTION_BADGE_ICON_CLASSNAME } from '$lib/utils';

	interface Props {
		class?: string;
		href?: string;
		name: string;
		path: string;
	}

	let { class: className = '', href, name, path }: Props = $props();

	// directories are encoded with a trailing `/` in the file:// target
	const Icon = $derived(path.endsWith('/') ? Folder : File);
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
		<span class="shrink-0 truncate">{name}</span>
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
		<span class="shrink-0 truncate">{name}</span>
	</span>
{/if}
