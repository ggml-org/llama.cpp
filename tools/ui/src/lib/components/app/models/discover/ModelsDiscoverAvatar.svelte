<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { DARK_INVERT_AVATAR_ORGS } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';

	interface Props {
		class?: string;
		org: string;
		quantOrg?: string;
		size?: string;
		baseImageClass?: string;
		quantImageClass?: string;
		quantPositionClass?: string;
		quantSize?: string;
	}

	let {
		baseImageClass = '',
		class: className = '',
		org,
		quantImageClass = 'h-full w-full',
		quantOrg,
		quantPositionClass = '-bottom-0.75 -right-0.75',
		quantSize = 'h-4.25 w-4.25',
		size = 'h-9 w-9'
	}: Props = $props();

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

<span class="relative mt-0.5 inline-flex shrink-0 {className}">
	{#if avatarError}
		<span
			aria-hidden="true"
			class="flex {size} items-center justify-center rounded-md text-sm font-semibold text-white"
			style="background-color: hsl({hue} 60% 45%)"
		>
			{org.charAt(0).toUpperCase()}
		</span>
	{:else}
		<div class="rounded-md">
			<img
				alt=""
				class="{size} rounded-md {invertAvatar ? 'dark:invert' : ''} {baseImageClass}"
				loading="lazy"
				onerror={() => (avatarError = true)}
				src={HuggingFaceService.getAvatarUrl(org)}
			/>
		</div>
	{/if}

	{#if quantOrg && quantOrg !== org}
		<Tooltip.Root>
			<Tooltip.Trigger
				class="absolute {quantPositionClass} {quantSize} overflow-hidden rounded-full border border-background bg-muted "
			>
				{#if quantError}
					<span
						aria-hidden="true"
						class="flex h-full w-full items-center justify-center rounded-full text-[8px] font-semibold text-white"
						style="background-color: hsl({quantHue} 60% 45%)"
					>
						{quantOrg.charAt(0).toUpperCase()}
					</span>
				{:else}
					<img
						alt=""
						class="{quantImageClass} rounded-full {invertQuant ? 'dark:invert' : ''}"
						loading="lazy"
						onerror={() => (quantError = true)}
						src={HuggingFaceService.getAvatarUrl(quantOrg)}
					/>
				{/if}
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p>{quantOrg}</p>
			</Tooltip.Content>
		</Tooltip.Root>
	{/if}
</span>
