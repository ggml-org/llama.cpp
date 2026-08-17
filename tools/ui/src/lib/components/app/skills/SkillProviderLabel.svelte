<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		GENERIC_SKILL_PROVIDER_TOOLTIP,
		skillProviderLabel
	} from './skill-provider-presentation';

	interface Props {
		provider: string;
	}

	let { provider }: Props = $props();

	const label = $derived(skillProviderLabel(provider));
	const isGenericProvider = $derived(provider === 'agents');
</script>

{#if isGenericProvider}
	<Tooltip.Provider><Tooltip.Root><Tooltip.Trigger>
		{#snippet child({ props })}<span {...props}>{label}</span>{/snippet}
	</Tooltip.Trigger><Tooltip.Content><p>{GENERIC_SKILL_PROVIDER_TOOLTIP}</p></Tooltip.Content></Tooltip.Root></Tooltip.Provider>
{:else}
	<span>{label}</span>
{/if}