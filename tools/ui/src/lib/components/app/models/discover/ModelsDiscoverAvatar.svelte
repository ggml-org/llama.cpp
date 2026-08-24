<script lang="ts">
	import { HuggingFaceService } from '$lib/services';
	import { DARK_INVERT_AVATAR_ORGS } from '$lib/constants';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		/** Org whose avatar is shown (may differ from the repo's org for base models). */
		org: string;
		/** Repo's own org, shown as a small corner badge when provided. */
		quantOrg?: string;
	}

	let { org, quantOrg }: Props = $props();

	let avatarError = $state(false);
	let quantError = $state(false);

	let invertAvatar = $derived(DARK_INVERT_AVATAR_ORGS.includes(org));
	let invertQuant = $derived(DARK_INVERT_AVATAR_ORGS.includes(quantOrg ?? ''));

	// Monogram fallback: org initial on a hue derived from its name, so each org
	// gets a stable distinct color.
	let hue = $derived.by(() => {
		let h = 0;

		for (let i = 0; i < org.length; i++) h = (h * 31 + org.charCodeAt(i)) >>> 0;

		return h % 360;
	});

	let quantHue = $derived.by(() => {
		const name = quantOrg ?? '';
		let h = 0;

		for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;

		return h % 360;
	});
</script>

<span class="relative mt-0.5 inline-flex shrink-0">
	{#if avatarError}
		<span
			class="flex h-9 w-9 items-center justify-center rounded-md text-sm font-semibold text-white"
			style="background-color: hsl({hue} 60% 45%)"
			aria-hidden="true"
		>
			{org.charAt(0).toUpperCase()}
		</span>
	{:else}
        <div class="rounded-md">
    		<img
    			src={HuggingFaceService.getAvatarUrl(org)}
    			onerror={() => (avatarError = true)}
    			class="h-9 w-9 rounded-md {invertAvatar ? 'dark:invert' : ''}"
    			alt=""
    			loading="lazy"
    		/>
	    </div>
	{/if}

	{#if quantOrg && quantOrg !== org}
		<Tooltip.Root>
			<Tooltip.Trigger
				class="absolute -bottom-0.75 -right-0.75 h-4.25 w-4.25 overflow-hidden rounded-full border border-background bg-muted "
			>
				{#if quantError}
					<span
						class="flex h-full w-full items-center justify-center rounded-full text-[8px] font-semibold text-white"
						style="background-color: hsl({quantHue} 60% 45%)"
						aria-hidden="true"
					>
						{quantOrg.charAt(0).toUpperCase()}
					</span>
				{:else}
					<img
						src={HuggingFaceService.getAvatarUrl(quantOrg)}
						onerror={() => (quantError = true)}
						class="h-full w-full rounded-full {invertQuant ? 'dark:invert' : ''}"
						alt=""
						loading="lazy"
					/>
				{/if}
			</Tooltip.Trigger>
			<Tooltip.Content>
				<p>{quantOrg}</p>
			</Tooltip.Content>
		</Tooltip.Root>
	{/if}
</span>
