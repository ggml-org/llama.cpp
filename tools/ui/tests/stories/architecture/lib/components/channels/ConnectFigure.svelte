<!--
	Llama Connect, as one scene: a laptop runs the Llama app, which serves llama-server and Llama Chat and shows a
	pairing code; a smartphone scans it from chat.llama.app, and the two then talk directly over WebRTC. The phone's
	prompts run on the laptop's model, and the laptop's conversations open on the phone, the same one highlighted on both.
-->
<script lang="ts">
	import Figure from '../ui/Figure.svelte';

	// a QR-like code, 21 modules square: the three finder squares, and the rest from a fixed seed so it never changes
	const N = 21;
	const finder = (x: number, y: number) =>
		[
			[0, 0],
			[N - 7, 0],
			[0, N - 7]
		].some(([fx, fy]) => x >= fx - 1 && x <= fx + 7 && y >= fy - 1 && y <= fy + 7);
	let seed = 7;
	const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
	const modules = Array.from({ length: N * N }, (_, i) => [i % N, Math.floor(i / N)]).filter(
		([x, y]) => !finder(x, y) && random() < 0.48
	);
	const QR = { size: 3.4, x: 333, y: 128 };

	const chats = ['Trip to Lisbon', 'Fix the CSV export', 'Birthday menu', 'Bike gears'];
</script>

<Figure
	label="A laptop runs the Llama app, which serves llama-server and Llama Chat and shows a pairing code. A smartphone opens chat.llama.app and scans the code. From then on the two talk directly over WebRTC: the phone's prompts run on the laptop's model, and the conversation Trip to Lisbon, started on the laptop, continues on the phone."
	minWidth={760}
>
	<svg viewBox="0 0 900 446" xmlns="http://www.w3.org/2000/svg">
		<defs>
			<marker
				id="connect-arrow"
				markerHeight="7"
				markerWidth="7"
				orient="auto-start-reverse"
				refX="9"
				refY="5"
				viewBox="0 0 10 10"><path class="ah" d="M0,0 L10,5 L0,10 z" /></marker
			>
		</defs>

		<!-- the laptop: lid, display, base -->
		<rect class="body" height="272" rx="14" width="440" x="30" y="30" />

		<rect class="screen" height="248" rx="4" width="416" x="42" y="42" />

		<path class="body" d="M14,302 H486 L474,320 H26 Z" />

		<path class="detail" d="M214,302 v6 h72 v-6" />

		<!-- Llama Chat on the laptop: the sidebar of conversations and the open one -->
		<rect class="pane" height="248" width="128" x="42" y="42" />

		<text class="ui-title" x="54" y="66">Llama Chat</text>

		{#each chats as chat, i (chat)}
			{#if i === 0}<rect
					class="current"
					height="24"
					rx="5"
					width="112"
					x="50"
					y={80 + i * 28}
				/>{/if}

			<text class={['ui', { on: i === 0 }]} x="58" y={96 + i * 28}>{chat}</text>
		{/each}

		<text class="ui-muted" x="54" y="278">Qwen3 8B · this laptop</text>

		<rect class="bubble" height="26" rx="8" width="150" x="196" y="58" />

		<text class="ui" x="206" y="75">Three days in Lisbon?</text>

		<path class="text-line" d="M188,108 H300 M188,124 H296 M188,140 H286 M188,156 H250" />

		<rect class="composer" height="28" rx="8" width="136" x="182" y="250" />

		<!-- the pairing code over the chat, as the Llama app shows it -->
		<rect class="card" height="168" rx="10" width="122" x="322" y="62" />

		<text class="ui-title" x="334" y="84">Pair a device</text>

		<rect
			class="qr-ground"
			height={N * QR.size + 8}
			width={N * QR.size + 8}
			x={QR.x - 4}
			y={QR.y - 4}
		/>

		{#each [[0, 0], [N - 7, 0], [0, N - 7]] as [fx, fy] (fx + '-' + fy)}
			<rect
				class="qr-ring"
				height={5.4 * QR.size}
				width={5.4 * QR.size}
				x={QR.x + (fx + 0.8) * QR.size}
				y={QR.y + (fy + 0.8) * QR.size}
			/>

			<rect
				class="qr"
				height={3 * QR.size}
				width={3 * QR.size}
				x={QR.x + (fx + 2) * QR.size}
				y={QR.y + (fy + 2) * QR.size}
			/>
		{/each}

		{#each modules as [x, y] (x + '-' + y)}
			<rect
				class="qr"
				height={QR.size}
				width={QR.size}
				x={QR.x + x * QR.size}
				y={QR.y + y * QR.size}
			/>
		{/each}

		<text class="code" x="334" y="220">K7Q-4ME</text>

		<!-- the smartphone -->
		<rect class="body" height="330" rx="28" width="168" x="700" y="20" />

		<rect class="screen" height="298" rx="18" width="148" x="710" y="36" />

		<rect class="notch" height="8" rx="4" width="44" x="762" y="44" />

		<text class="ui-muted" x="722" y="74">chat.llama.app</text>

		<rect class="current" height="26" rx="6" width="124" x="722" y="84" />

		<text class="ui on" x="732" y="101">‹ Trip to Lisbon</text>

		<rect class="bubble" height="26" rx="8" width="112" x="734" y="120" />

		<text class="ui" x="742" y="137">Three days in…</text>

		<path class="text-line" d="M722,170 H830 M722,186 H824 M722,202 H790" />

		<rect class="bubble" height="26" rx="8" width="96" x="750" y="218" />

		<text class="ui" x="758" y="235">And a day trip?</text>

		<path class="text-line typing" d="M722,262 H800" />

		<rect class="typing-cursor" height="12" width="3" x="804" y="256" />

		<rect class="composer" height="28" rx="8" width="124" x="722" y="294" />

		<!-- 1 the code, 2 the scan, 3 the direct link -->
		<path class="scan" d="M708,64 C620,8 520,24 448,104" marker-end="url(#connect-arrow)" />

		<circle class="step" cx="496" cy="24" r="11" />

		<text class="step-n" x="496" y="28">2</text>

		<text class="note" x="514" y="28">The phone scans the code</text>

		<circle class="step" cx="383" cy="246" r="11" />

		<text class="step-n" x="383" y="250">1</text>

		<path
			class="link"
			d="M494,196 H692"
			marker-end="url(#connect-arrow)"
			marker-start="url(#connect-arrow)"
		/>

		<circle class="step" cx="593" cy="160" r="11" />

		<text class="step-n" x="593" y="164">3</text>

		<text class="via" text-anchor="middle" x="593" y="186">WebRTC, direct</text>

		<text class="note" text-anchor="middle" x="593" y="220">prompts in, replies out,</text>

		<text class="note" text-anchor="middle" x="593" y="236">and the conversations</text>

		<!-- what each device is -->
		<text class="t" x="30" y="374">Laptop</text>

		<text class="s" x="30" y="396"
			>The Llama app runs llama-server with --connect and the model;</text
		>

		<text class="s" x="30" y="414"
			>Llama Chat keeps the conversations here, and shows the code (1).</text
		>

		<text class="t" x="700" y="374">Smartphone</text>

		<text class="s" x="700" y="396">chat.llama.app, the UI only;</text>

		<text class="s" x="700" y="414">the model and the chats</text>

		<text class="s" x="700" y="432">stay on the laptop</text>
	</svg>
</Figure>

<style>
	text {
		font-family: var(--sans);
		fill: var(--fg);
	}
	.t {
		font-size: 15px;
		font-weight: 600;
	}
	.s,
	.note {
		font-size: 12.5px;
		fill: var(--muted);
	}
	.via {
		font-family: var(--mono);
		font-size: 12px;
		font-weight: 500;
		fill: var(--app);
	}
	.code {
		font-family: var(--mono);
		font-size: 13px;
		letter-spacing: 0.08em;
	}

	/* the devices: a raised body, the display in the page's colour */
	.body {
		fill: var(--surface);
		stroke: var(--fg);
		stroke-width: 1.8;
		stroke-linejoin: round;
	}
	.screen {
		fill: var(--bg);
		stroke: var(--line);
	}
	.notch {
		fill: var(--fg);
		opacity: 0.8;
	}
	.detail {
		fill: none;
		stroke: var(--fg);
		stroke-width: 1.4;
		opacity: 0.4;
	}

	/* Llama Chat, small: the sidebar, bubbles, lines of a reply, the composer */
	.pane {
		fill: var(--be-bg);
	}
	.ui {
		font-size: 11.5px;
	}
	.ui.on {
		font-weight: 600;
		fill: var(--app);
	}
	.ui-title {
		font-size: 12px;
		font-weight: 600;
	}
	.ui-muted {
		font-family: var(--mono);
		font-size: 10.5px;
		fill: var(--muted);
	}
	.current {
		fill: var(--app-bg);
		stroke: var(--app);
		stroke-width: 1.2;
	}
	.bubble {
		fill: var(--be-bg);
	}
	.text-line {
		fill: none;
		stroke: var(--fg);
		stroke-width: 5;
		stroke-linecap: round;
		opacity: 0.18;
	}
	.typing-cursor {
		fill: var(--art-cinder);
	}
	.composer {
		fill: none;
		stroke: var(--line);
		stroke-width: 1.4;
	}

	/* the pairing card and its code, dark on white in either theme so it reads as a code to scan */
	.card {
		fill: var(--surface);
		stroke: var(--fg);
		stroke-width: 1.4;
	}
	.qr-ground {
		fill: white;
	}
	.qr {
		fill: #111;
	}
	.qr-ring {
		fill: none;
		stroke: #111;
		stroke-width: 3.4;
	}

	/* the steps: the scan dotted, the direct link dashed in cinder */
	.scan {
		fill: none;
		stroke: var(--muted);
		stroke-width: 1.6;
		stroke-dasharray: 2 5;
		stroke-linecap: round;
	}
	.link {
		fill: none;
		stroke: var(--app);
		stroke-width: 2;
		stroke-dasharray: 7 6;
	}
	.ah {
		fill: var(--muted);
	}
	.step {
		fill: var(--fg);
	}
	.step-n {
		font-family: var(--mono);
		font-size: 12px;
		font-weight: 600;
		fill: var(--bg);
		text-anchor: middle;
	}
</style>
