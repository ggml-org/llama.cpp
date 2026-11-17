// File mention badges for the chat form input: in the block being edited, each
// `[name](file://path)` link shows as the same chip the rendered markdown draws.
// The link stays in the source untouched; the badge is an atomic stand-in for it.

import type { EditorState, Range } from '@codemirror/state';
import { Decoration, WidgetType } from '@codemirror/view';
import { codeRanges } from '$lib/components/app/content/MarkdownEditor/markdown-syntax';
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

const MENTION_RE = new RegExp(fileMentionLinkRe().source, 'g');

class MentionBadgeWidget extends WidgetType {
	readonly decodedPath: string;
	readonly label: string;

	constructor(
		readonly name: string,
		readonly path: string
	) {
		super();
		this.decodedPath = decodeFileLinkPath(path);
		this.label = getMentionBadgeLabel(
			name,
			this.decodedPath,
			settingsStore.getConfig(SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS),
			toolsStore.serverHome
		);
	}

	// the label follows the mention settings, so a changed setting redraws the badge
	eq(other: MentionBadgeWidget) {
		return other.path === this.path && other.label === this.label;
	}

	toDOM() {
		const badge = document.createElement('span');

		badge.title = this.decodedPath;
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
		label.textContent = this.label;

		badge.append(svg, label);

		return badge;
	}
}

/**
 * Badge widgets for the mention links between `from` and `to`. Code is literal: a
 * mention link in a code block or in inline code stays text.
 */
export function fileMentionWidgets(
	state: EditorState,
	from: number,
	to: number
): Range<Decoration>[] {
	const code = codeRanges(state, from, to);
	const widgets: Range<Decoration>[] = [];

	for (const match of state.doc.sliceString(from, to).matchAll(MENTION_RE)) {
		const start = from + match.index;
		const end = start + match[0].length;

		if (code.some(([codeFrom, codeTo]) => start < codeTo && end > codeFrom)) continue;

		const widget = new MentionBadgeWidget(match[1], match[2]);

		widgets.push(Decoration.replace({ widget }).range(start, end));
	}

	return widgets;
}
