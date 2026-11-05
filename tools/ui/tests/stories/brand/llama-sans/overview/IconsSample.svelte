<!--
	Llama Sans UI with Lucide icons, the design system's icon set: a toolbar, a menu and a sidebar list, all on the
	font's own line box, then the icons on their own. The line box is centred on the cap height, so an icon centred in
	it lines up with the capitals and figures beside it, with no nudging.
-->
<script lang="ts">
	import {
		Bell,
		Bot,
		CircleAlert,
		CircleCheck,
		Code,
		Copy,
		Cpu,
		Database,
		Download,
		Ellipsis,
		FileText,
		Folder,
		FolderOpen,
		Gauge,
		GitBranch,
		GitCommitHorizontal,
		Globe,
		HardDrive,
		Image,
		Keyboard,
		Layers,
		Link,
		Lock,
		MessageSquare,
		MessageSquarePlus,
		Mic,
		Paperclip,
		Play,
		Search,
		Server,
		Settings,
		Sparkles,
		Square,
		Terminal,
		Trash,
		Upload,
		Wrench,
		Zap
	} from '@lucide/svelte';

	const menu = [
		{ icon: MessageSquarePlus, keys: '⌘N', label: 'New chat' },
		{ icon: Search, keys: '⌘K', label: 'Search chats' },
		{ icon: FolderOpen, keys: '⌘O', label: 'Open model…' },
		{ icon: Cpu, keys: '', label: 'Hardware' },
		{ icon: Keyboard, keys: '?', label: 'Keyboard shortcuts' },
		{ icon: Settings, keys: '⌘,', label: 'Settings' }
	];
	const models = [
		{ name: 'Qwen 2.5 7B Instruct', size: '4.68 GB', status: 'Loaded', tone: 'ok' },
		{ name: 'Llama 3.2 3B', size: '2.02 GB', status: 'Ready', tone: '' },
		{ name: 'Gemma 2 27B', size: '16.6 GB', status: 'Too large', tone: 'warn' }
	];
	const grid = [
		Play,
		Square,
		Download,
		Upload,
		Settings,
		Search,
		MessageSquare,
		Bot,
		Cpu,
		HardDrive,
		Server,
		Zap,
		Gauge,
		Layers,
		Database,
		FileText,
		Folder,
		Image,
		Mic,
		Paperclip,
		Link,
		Lock,
		Bell,
		Globe,
		Terminal,
		Code,
		GitBranch,
		GitCommitHorizontal,
		Wrench,
		Sparkles,
		Copy,
		Trash
	];
</script>

<div class="icons-sample">
	<div aria-label="Model" class="toolbar" role="toolbar">
		<span class="button primary"><Play size={16} strokeWidth={1.75} />Run</span>

		<span class="button"><Square size={16} strokeWidth={1.75} />Stop</span>

		<span class="button"><Download size={16} strokeWidth={1.75} />Pull model</span>

		<span class="sep"></span>

		<span aria-label="Settings" class="button only"><Settings size={16} strokeWidth={1.75} /></span>

		<span aria-label="More" class="button only"><Ellipsis size={16} strokeWidth={1.75} /></span>
	</div>

	<div class="pair">
		<ul aria-label="Menu" class="menu">
			{#each menu as m (m.label)}
				<li><m.icon size={16} strokeWidth={1.75} /><span>{m.label}</span><kbd>{m.keys}</kbd></li>
			{/each}
		</ul>

		<ul aria-label="Models" class="list">
			{#each models as m (m.name)}
				<li>
					<HardDrive size={16} strokeWidth={1.75} />

					<span class="name">{m.name}<small>{m.size} · Q4_K_M</small></span>

					<span class="chip {m.tone}">
						{#if m.tone === 'ok'}<CircleCheck
								size={12}
								strokeWidth={2}
							/>{:else if m.tone === 'warn'}<CircleAlert size={12} strokeWidth={2} />{/if}{m.status}
					</span>
				</li>
			{/each}
		</ul>
	</div>

	<figure class="grid">
		<div aria-hidden="true" class="glyphs">
			{#each grid as Icon, i (i)}<span><Icon size={24} strokeWidth={1.5} /></span>{/each}
		</div>

		<figcaption>
			Lucide at 1.5 to 2 px strokes, the same even weight as the letters. Next to UI text, an icon
			centred in the line box sits on the cap height, so buttons, menus and chips need no per-icon
			nudging.
		</figcaption>
	</figure>
</div>

<style>
	.icons-sample {
		display: grid;
		gap: 20px;
		font-family: var(--ui);
	}
	.toolbar {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
		padding: 6px;
		border: 1px solid var(--rule);
		border-radius: 12px;
		background: var(--sheet);
		font: 500 14px / normal var(--ui);
	}
	.button {
		display: inline-flex;
		gap: 6px;
		align-items: center;
		padding: 7px 12px;
		border-radius: 8px;
		background: var(--paper);
		box-shadow: inset 0 0 0 1px var(--rule);
	}
	.button.only {
		padding: 7px;
	}
	.primary {
		background: var(--ink);
		color: var(--paper);
		box-shadow: none;
	}
	.sep {
		width: 1px;
		height: 20px;
		margin-inline: 4px;
		background: var(--rule);
	}
	.pair {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr));
		gap: 16px;
	}
	ul {
		display: grid;
		gap: 2px;
		margin: 0;
		padding: 6px;
		border: 1px solid var(--rule);
		border-radius: 12px;
		background: var(--sheet);
		list-style: none;
		font: 400 14px / normal var(--ui);
	}
	li {
		display: flex;
		gap: 10px;
		align-items: center;
		padding: 7px 10px;
		border-radius: 8px;
	}
	.menu li:first-child {
		background: var(--paper);
		box-shadow: inset 0 0 0 1px var(--rule);
	}
	.menu span {
		flex: 1;
	}
	kbd {
		font: 400 12px / normal var(--ui);
		color: var(--muted);
	}
	.list .name {
		display: grid;
		flex: 1;
		font-weight: 500;
	}
	.list small {
		font: 400 12px / normal var(--ui);
		color: var(--muted);
	}
	.chip {
		display: inline-flex;
		gap: 4px;
		align-items: center;
		padding: 3px 8px;
		border-radius: 999px;
		background: var(--paper);
		box-shadow: inset 0 0 0 1px var(--rule);
		font: 500 12px / normal var(--ui);
	}
	.chip.ok {
		color: var(--ink);
	}
	.chip.warn {
		color: var(--status-warning-glyph); /* the UI's own warning colour */
	}
	.grid {
		display: grid;
		gap: 12px;
		margin: 0;
	}
	.glyphs {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(44px, 1fr));
		gap: 4px;
	}
	.glyphs span {
		display: grid;
		place-items: center;
		aspect-ratio: 1;
		border-radius: 10px;
		color: var(--ink);
	}
	.glyphs span:hover {
		background: var(--sheet);
	}
	figcaption {
		max-width: 60ch;
		font: 400 13px / 1.5 var(--ui);
		color: var(--muted);
	}
</style>
