/**
 * Records a page through Chromium's screencast instead of Playwright's video,
 * which encodes VP8 at about 1 Mbit/s at 1x scale. Frames arrive at the page's
 * device pixel ratio, each with its timestamp, and are encoded once into an
 * H.264 mp4 (what GitHub accepts in PR descriptions).
 */

import type { CDPSession, Page } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const FPS = 60;
const JPEG_QUALITY = 92;

interface Frame {
	file: string;
	/** Seconds, on the same clock as Date.now(). */
	time: number;
}

export class ScreencastRecorder {
	private readonly dir: string;
	private readonly frames: Frame[] = [];
	private readonly pending: Promise<void>[] = [];
	private session: CDPSession | undefined;
	private stoppedAt = 0;

	constructor(dir: string) {
		this.dir = dir;
	}

	/** Encodes the frames, each held until the next one arrived (the screencast only sends changes). */
	encode(mp4: string): void {
		if (this.frames.length === 0) throw new Error('demo: the screencast captured no frames');

		const list = this.frames.flatMap((frame, i) => {
			const end = this.frames[i + 1]?.time ?? this.stoppedAt;

			return [`file '${frame.file}'`, `duration ${Math.max(end - frame.time, 1 / FPS).toFixed(4)}`];
		});
		const listFile = join(this.dir, 'frames.txt');

		// the concat demuxer ignores the last entry's duration unless the file is repeated
		writeFileSync(listFile, [...list, `file '${this.frames.at(-1)!.file}'`, ''].join('\n'));
		execFileSync(
			'ffmpeg',
			[
				'-y',
				'-loglevel',
				'error',
				'-f',
				'concat',
				'-safe',
				'0',
				'-i',
				listFile,
				'-vf',
				`fps=${FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2`,
				'-c:v',
				'libx264',
				'-preset',
				'slow',
				'-crf',
				'18',
				'-pix_fmt',
				'yuv420p',
				'-movflags',
				'+faststart',
				mp4
			],
			{ stdio: 'inherit' }
		);
		rmSync(this.dir, { force: true, recursive: true });
	}

	async start(page: Page): Promise<void> {
		const size = page.viewportSize();
		const scale = await page.evaluate(() => window.devicePixelRatio);

		rmSync(this.dir, { force: true, recursive: true });
		mkdirSync(this.dir, { recursive: true });

		this.session = await page.context().newCDPSession(page);
		this.session.on('Page.screencastFrame', ({ data, metadata, sessionId }) => {
			const file = join(this.dir, `${String(this.frames.length).padStart(6, '0')}.jpg`);

			// acknowledged first: Chromium sends the next frame only after the ack
			this.session?.send('Page.screencastFrameAck', { sessionId }).catch(() => {});
			this.frames.push({ file, time: metadata.timestamp ?? Date.now() / 1000 });
			this.pending.push(writeFile(file, Buffer.from(data, 'base64')));
		});
		await this.session.send('Page.startScreencast', {
			everyNthFrame: 1,
			format: 'jpeg',
			maxHeight: size ? size.height * scale : undefined,
			maxWidth: size ? size.width * scale : undefined,
			quality: JPEG_QUALITY
		});
	}

	async stop(): Promise<void> {
		this.stoppedAt = Date.now() / 1000;
		await this.session?.send('Page.stopScreencast').catch(() => {});
		await this.session?.detach().catch(() => {});
		await Promise.all(this.pending);
	}
}
