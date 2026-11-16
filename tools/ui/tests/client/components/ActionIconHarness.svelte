<script lang="ts">
	import { ActionIcon } from '$lib/components/app';
	import { CodeBlockActions } from '$lib/components/app/misc';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { Component } from 'svelte';

	interface Props {
		code?: string;
		disabled?: boolean;
		icon?: Component;
		language?: string;
		testCodeBlockActions?: boolean;
		tooltip?: string;
		triggerClass?: string;
	}

	let {
		code = '',
		disabled = false,
		icon,
		language = 'bash',
		testCodeBlockActions = false,
		tooltip,
		triggerClass
	}: Props = $props();
</script>

<Tooltip.Provider>
	{#if testCodeBlockActions}
		<CodeBlockActions {code} {disabled} {language} />
	{:else if triggerClass}
		<Tooltip.Root>
			<Tooltip.Trigger class={triggerClass}>
				<span>Trigger</span>
			</Tooltip.Trigger>
		</Tooltip.Root>
	{:else if icon}
		<ActionIcon {disabled} {icon} {tooltip} />
	{/if}
</Tooltip.Provider>
