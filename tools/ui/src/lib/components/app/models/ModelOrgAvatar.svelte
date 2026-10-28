<script lang="ts">
	import { DARK_INVERT_AVATAR_ORGS } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';
	import { SvelteMap } from 'svelte/reactivity';

	// Avatars a failed request is cooling down for. A failure hides the image
	// immediately, but the org gets another chance on a later mount: the cooldown
	// expires on its own, so a transient network blip is not permanent for the
	// session. A row remounting inside the cooldown renders the monogram instead
	// of re-requesting the same failing URL.
	const AVATAR_RETRY_DELAY_MS = 60_000;
	const failedAvatarOrgs = new SvelteMap<string, number>();

	/** True while an org's failed avatar is still inside its retry cooldown. */
	function isAvatarCoolingDown(org: string): boolean {
		const failedAt = failedAvatarOrgs.get(org);

		// a stale entry reads as not failed and is dropped on the next onerror
		return failedAt !== undefined && Date.now() - failedAt < AVATAR_RETRY_DELAY_MS;
	}

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

	// With the Hub metadata setting off there is nothing to fetch and nothing to
	// show: the avatar is hidden entirely instead of falling back to a monogram.
	let hubEnabled = $derived(HuggingFaceService.isEnabled());
	let orgAvatarFailed = $derived(isAvatarCoolingDown(org));
	let quantAvatarFailed = $derived(isAvatarCoolingDown(quantOrg ?? ''));

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

{#if hubEnabled}
	<span class="relative inline-flex shrink-0 {className}">
		{#if orgAvatarFailed}
			<span
				aria-hidden="true"
				class="flex {size} items-center justify-center rounded-md text-sm font-semibold text-white"
				style="background-color: hsl({hue} 60% 45%)"
			>
				{org.charAt(0).toUpperCase()}
			</span>
		{:else}
			<div class="rounded-md">
				<!-- the server serves the app under COEP require-corp, so a cross-origin
				     image has to be fetched in CORS mode to be allowed -->
				<img
					alt=""
					class="{size} rounded-md {invertAvatar ? 'dark:invert' : ''} {baseImageClass}"
					crossorigin="anonymous"
					loading="lazy"
					onerror={() => {
						failedAvatarOrgs.set(org, Date.now());
					}}
					src={HuggingFaceService.getAvatarUrl(org)}
				/>
			</div>
		{/if}

		{#if quantOrg && quantOrg !== org}
			<!-- native title instead of a floating tooltip: long model lists mount one badge per row -->
			<span
				class="absolute {quantPositionClass} {quantSize} overflow-hidden rounded-full border border-background bg-muted"
				title={quantOrg}
			>
				{#if quantAvatarFailed}
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
						crossorigin="anonymous"
						loading="lazy"
						onerror={() => {
							failedAvatarOrgs.set(quantOrg ?? '', Date.now());
						}}
						src={HuggingFaceService.getAvatarUrl(quantOrg)}
					/>
				{/if}
			</span>
		{/if}
	</span>
{/if}
