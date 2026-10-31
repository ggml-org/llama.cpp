// File mention badges for the chat form input: in the block being edited, each
// `[name](file://path)` link shows as the same chip the rendered markdown draws.
// The link stays in the source untouched; the badge is an atomic stand-in for it.

import { type Range } from '@codemirror/state';
import { Decoration, WidgetType } from '@codemirror/view';
import { SETTINGS_KEYS } from '$lib/constants';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import {
	decodeFileLinkPath,
	fileMentionLinkRe,
	getMentionBadgeIconPaths,
	getMentionBadgeLabel,
	MENTION_BADGE_CLASSNAME,
	MENTION_BADGE_ICON_CLASSNAME,
	MENTION_BADGE_SVG_ATTRIBUTES
} from '$lib/utils';

// Inline code is literal: a mention link inside backticks stays text
const CODE_OR_MENTION_RE = new RegExp(
	String.raw`(\`\`\`[\s\S]*?\`\`\`|\`[^\`\n]+\`)|${fileMentionLinkRe().source}`,
	'g'
);

class MentionBadgeWidget extends WidgetType {
	constructor(
		readonly name: string,
		readonly path: string
	) {
		super();
	}

	eq(other: MentionBadgeWidget) {
		return other.name === this.name && other.path === this.path;
	}

	toDOM() {
		const badge = document.createElement('span');
		const decodedPath = decodeFileLinkPath(this.path);

		badge.title = decodedPath;
		badge.className = `${MENTION_BADGE_CLASSNAME} align-middle`;

		const svg = document.createElementNS(MENTION_BADGE_SVG_ATTRIBUTES['xmlns'], 'svg');

		for (const [attr, value] of Object.entries(MENTION_BADGE_SVG_ATTRIBUTES)) {
			svg.setAttribute(attr, value);
		}

		for (const cls of MENTION_BADGE_ICON_CLASSNAME.split(/\s+/).filter(Boolean)) {
			svg.classList.add(cls);
		}

		for (const d of getMentionBadgeIconPaths(this.path)) {
			const path = document.createElementNS(MENTION_BADGE_SVG_ATTRIBUTES['xmlns'], 'path');

			path.setAttribute('d', d);
			svg.appendChild(path);
		}

		const label = document.createElement('span');

		label.classList.add('shrink-0', 'truncate');
		label.textContent = getMentionBadgeLabel(
			this.name,
			decodedPath,
			settingsStore.getConfig(SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS),
			toolsStore.serverHome
		);

		badge.append(svg, label);

		return badge;
	}
}

/** Badge widgets for the mention links in `source`, which starts at document offset `from`. */
export function fileMentionWidgets(source: string, from: number): Range<Decoration>[] {
	const widgets: Range<Decoration>[] = [];

	for (const match of source.matchAll(CODE_OR_MENTION_RE)) {
		if (match[1] !== undefined) continue;

		const start = from + match.index;
		const widget = new MentionBadgeWidget(match[2], match[3]);

		widgets.push(Decoration.replace({ widget }).range(start, start + match[0].length));
	}

	return widgets;
}
