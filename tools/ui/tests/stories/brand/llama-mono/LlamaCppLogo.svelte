<!--
	The llama.cpp logo, Llama Mono's own lockup: the README cover as on github.com/ggml-org/llama.cpp, the lockup files,
	how it's typed (U+E002, then llama.cpp in lowercase Llama Mono Bold, .cpp in cinder), the sizes it's set at, and
	what it never does. Everything but the files is typed, so it shows what the font gives anyone who sets it.
-->
<script lang="ts">
	import dark from '../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-dark.svg?url';
	import darkBg from '../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-dark-bg.svg?url';
	import light from '../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-light.svg?url';
	import lightBg from '../assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-light-bg.svg?url';

	const LOCKUP = '';

	const files = [
		{ ground: 'dark', name: 'logo-llama-cpp-dark-bg', src: darkBg },
		{ ground: 'light', name: 'logo-llama-cpp-light-bg', src: lightBg },
		{ ground: 'dark', name: 'logo-llama-cpp-dark', src: dark },
		{ ground: 'light', name: 'logo-llama-cpp-light', src: light }
	];

	const sizes = [64, 40, 24, 16, 12];

	const parts = [
		{
			label: 'U+E002',
			note: 'the icon with the gap built in, drawn 1/0.9 larger than in Llama Sans'
		},
		{ label: 'llama', note: 'Llama Mono Bold, always lowercase' },
		{ label: '.cpp', note: 'cinder, #EE6C1E, the brand’s one accent orange' }
	];
</script>

<!-- the README cover, 3:1, the lockup at 7.1% of its width -->
<div class="covers">
	{#each ['dark', 'light'] as const as tone (tone)}
		<div class="cover {tone}">
			<span class="mark">{LOCKUP}llama<span class="cpp">.cpp</span></span>
		</div>
	{/each}
</div>

<section class="block">
	<h2>Typed</h2>

	<div class="anatomy">
		<p class="mark big">
			<span class="part">{LOCKUP}</span><span class="part">llama</span><span class="part cpp"
				>.cpp</span
			>
		</p>

		<ol>
			{#each parts as p (p.label)}
				<li><code>{p.label}</code><span>{p.note}</span></li>
			{/each}
		</ol>
	</div>
</section>

<section class="block">
	<h2>The files</h2>

	<div class="files">
		{#each files as f (f.name)}
			<figure class="file {f.ground}">
				<img alt="The llama.cpp lockup, {f.name}" src={f.src} />

				<figcaption>{f.name}.svg</figcaption>
			</figure>
		{/each}
	</div>
</section>

<section class="block">
	<h2>Sizes</h2>

	<div class="sizes">
		{#each sizes as size (size)}
			<p>
				<span class="label">{size} px</span>

				<span style:font-size="{size}px" class="mark"
					>{LOCKUP}llama<span class="cpp">.cpp</span></span
				>
			</p>
		{/each}
	</div>
</section>

<section class="block">
	<h2>Never</h2>

	<div class="donts">
		<figure>
			<p class="mark">{LOCKUP}LLAMA<span class="cpp">.CPP</span></p>

			<figcaption>In capitals: it's a command, typed as one</figcaption>
		</figure>

		<figure>
			<p class="mark sans">{LOCKUP}llama<span class="cpp">.cpp</span></p>

			<figcaption>In Llama Sans: that's the app's lockup, Llama</figcaption>
		</figure>

		<figure class="on-colour">
			<p class="mark">{LOCKUP}llama<span class="cpp">.cpp</span></p>

			<figcaption>On colour: llama.cpp stays on neutral grounds</figcaption>
		</figure>
	</div>
</section>

<style>
	.mark {
		font-family: var(--code);
		font-weight: 700;
		white-space: nowrap;
	}
	.cpp {
		color: #ee6c1e;
	}
	h2 {
		margin: 0;
		font: 600 16px / 1.5 var(--ui);
	}
	.block {
		display: grid;
		gap: 16px;
	}

	.covers {
		display: grid;
		gap: 16px;
	}
	.cover {
		display: grid;
		place-items: center;
		aspect-ratio: 3;
		container-type: inline-size;
		border-radius: 10px;
	}
	.cover .mark {
		font-size: 7.1cqw;
	}
	.dark {
		background: #111111;
		color: #fafafa;
	}
	.light {
		background: #fafafa;
		color: #111111;
		box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.1);
	}

	.anatomy {
		display: grid;
		gap: 24px;
		padding: clamp(24px, 4vw, 48px);
		border-radius: 10px;
		background: var(--sheet);
	}
	.big {
		margin: 0;
		font-size: clamp(36px, 8vw, 96px);
		line-height: 1.2;
	}
	.part {
		box-shadow: inset 0 0 0 1px var(--guide-soft);
	}
	ol {
		display: grid;
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	li {
		display: grid;
		grid-template-columns: 7rem 1fr;
		gap: 16px;
		font: 400 15px / 1.5 var(--ui);
	}
	li code {
		font: 600 14px / 1.6 var(--code);
	}

	.files {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
		gap: 16px;
	}
	.file {
		display: grid;
		gap: 0;
		margin: 0;
		overflow: hidden;
		border-radius: 10px;
	}
	.file img {
		display: block;
		width: 100%;
		height: auto;
		padding: 12% 10%;
		box-sizing: border-box;
	}
	/* the -bg files bring their own ground */
	.file:nth-child(-n + 2) img {
		padding: 0;
	}
	.file figcaption {
		padding: 8px 12px;
		font: 500 12px var(--code);
		opacity: 0.6;
	}

	.sizes {
		display: grid;
		gap: 12px;
	}
	.sizes p {
		display: grid;
		grid-template-columns: 4rem 1fr;
		align-items: center;
		gap: 16px;
		margin: 0;
		overflow: hidden;
	}
	.sizes .label {
		font: 500 12px var(--code);
		color: var(--muted);
	}

	.donts {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr));
		gap: 16px;
	}
	.donts figure {
		display: grid;
		gap: 12px;
		margin: 0;
		padding: 28px 20px 16px;
		border-radius: 10px;
		background: var(--sheet);
		position: relative;
	}
	.donts figure::after {
		content: '';
		position: absolute;
		inset: 12px 12px auto auto;
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: var(--cinder);
		mask:
			linear-gradient(45deg, transparent 44%, #000 44% 56%, transparent 56%),
			linear-gradient(-45deg, transparent 44%, #000 44% 56%, transparent 56%);
	}
	.donts .mark {
		margin: 0;
		font-size: clamp(22px, 3vw, 30px);
		text-align: center;
	}
	.donts .sans {
		font-family: var(--headline);
	}
	.donts .on-colour {
		background: var(--llama-hero);
		color: var(--llama-hero-ink);
	}
	.donts figcaption {
		font: 400 13px / 1.4 var(--ui);
		color: var(--muted);
	}
	.donts .on-colour figcaption {
		color: inherit;
		opacity: 0.7;
	}
</style>
