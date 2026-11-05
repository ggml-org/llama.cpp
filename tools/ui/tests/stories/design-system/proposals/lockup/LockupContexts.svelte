<!--
	Proposal: steps 3 and 4 of the lockup side by side where a lockup is used, at the real sizes: an app header, a
	footer, a hero and the X header banner, so the proportion is judged small and large, not only at one size.
-->
<script lang="ts">
	import { step } from './geometry';
	import LockupMark from './LockupMark.svelte';
	import { Button } from '$lib/components/ui/button';

	const steps = [step(3), step(4)];

	const HERO =
		'radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%), linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%)';
</script>

<div class="mx-auto grid w-[min(72rem,calc(100%-2rem))] gap-12 py-10">
	<header class="grid gap-2">
		<h1 class="text-3xl font-bold tracking-tight">In context</h1>

		<p class="max-w-[64ch] text-sm text-muted-foreground">
			Step 3 (the icon 1.36× the cap height) and step 4 (1.18×) where the lockup is used, at real
			sizes. A proportion can hold up large and fall apart small, or the other way round.
		</p>
	</header>

	{#snippet pair(
		title: string,
		note: string,
		body: (s: (typeof steps)[number]) => ReturnType<import('svelte').Snippet>
	)}
		<section class="grid gap-3">
			<div class="grid gap-0.5">
				<h2 class="text-lg font-semibold">{title}</h2>

				<p class="text-xs text-muted-foreground">{note}</p>
			</div>

			<div class="grid gap-4 lg:grid-cols-2">
				{#each steps as s (s.label)}
					<figure class="m-0 grid gap-2">
						{@render body(s)}

						<figcaption class="font-mono text-xs text-muted-foreground">
							{s.label} · icon {s.icon.toFixed(2)}× cap
						</figcaption>
					</figure>
				{/each}
			</div>
		</section>
	{/snippet}

	{#snippet appHeader(s: (typeof steps)[number])}
		<div class="grid gap-2">
			{#each [{ bg: 'bg-background', ink: 'var(--foreground)' }] as t (t.bg)}
				<div
					class="flex h-14 items-center justify-between gap-4 rounded-card border border-border/60 {t.bg} px-4"
				>
					<LockupMark ink={t.ink} name="llama.cpp" size={18} step={s} />

					<nav class="flex items-center gap-4 text-body-sm text-muted-foreground max-sm:hidden">
						<span>Models</span><span>Docs</span><span>GitHub</span>
					</nav>

					<Button size="sm">Download</Button>
				</div>
			{/each}

			<div
				class="flex h-12 items-center gap-3 rounded-card border border-border/60 bg-[#111111] px-4 text-white"
			>
				<LockupMark ink="#fafafa" name="Llama" size={16} step={s} />

				<span class="text-body-sm text-white/60">Settings</span>
			</div>
		</div>
	{/snippet}

	{#snippet footer(s: (typeof steps)[number])}
		<div
			class="flex flex-wrap items-center justify-between gap-3 rounded-card border border-border/60 bg-surface-muted/40 px-4 py-5"
		>
			<LockupMark ink="var(--foreground)" name="llama.cpp" size={13} step={s} />

			<span class="text-caption text-muted-foreground">MIT licence · ggml-org</span>
		</div>
	{/snippet}

	{#snippet hero(s: (typeof steps)[number])}
		<div
			style:background={HERO}
			class="grid min-h-56 place-items-center rounded-card border border-border/60 p-6"
		>
			<LockupMark ink="#0f1f33" name="Llama" size={64} step={s} />
		</div>
	{/snippet}

	{#snippet banner(s: (typeof steps)[number])}
		<!-- the X header, 1500 × 500, with the lockup at the size build_social.py gives it -->
		<div
			class="grid aspect-[3/1] place-items-center rounded-card border border-border/60 bg-[#111111] [container-type:inline-size]"
		>
			<div class="w-[46cqi]">
				<LockupMark ink="#fafafa" name="llama.cpp" size={36} step={s} />
			</div>
		</div>
	{/snippet}

	{@render pair(
		'App header',
		'The lockup at 16 and 18 px, next to navigation and a button',
		appHeader
	)}

	{@render pair('Footer', 'The smallest use: 13 px', footer)}

	{@render pair('Hero', '64 px on the hero blue', hero)}

	{@render pair('X header', 'A social banner, 1500 × 500, at about a quarter of its size', banner)}
</div>
