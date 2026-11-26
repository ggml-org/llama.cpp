// Guards the media an `@` mention attaches: the same file mentioned twice while its first read
// is on its way is attached once, and a mention's attachment leaves with it even when another
// mention's path starts the same way.

import { FILE_URI_PREFIX } from '#lib/constants/index.ts';
import { useChatScreenFileUpload } from '#lib/hooks/use-chat-screen-file-upload.svelte.ts';
import { ReadMediaService } from '#lib/services/read-media.service.ts';
import { encodeFileLinkPath } from '#lib/utils/index.ts';
import { afterEach, expect, it, vi } from 'vitest';

const SHOT = '/Users/demo/Desktop/shot.png';
const SHOT_BACKUP = '/Users/demo/Desktop/shot.png.png';
const link = (path: string) => `[file](${FILE_URI_PREFIX}${encodeFileLinkPath(path)}) `;

function setup() {
	let upload!: ReturnType<typeof useChatScreenFileUpload>;

	const cleanup = $effect.root(() => {
		upload = useChatScreenFileUpload({
			activeModelId: () => null,
			capabilities: () => ({ hasAudio: false, hasVideo: false, hasVision: true })
		});
	});

	return { cleanup, upload };
}

afterEach(() => {
	vi.restoreAllMocks();
});

it('attaches a file mentioned twice during its read once', async () => {
	let finish!: (file: File) => void;

	const read = vi
		.spyOn(ReadMediaService, 'readFile')
		.mockImplementation(() => new Promise((resolve) => (finish = resolve)));
	const { cleanup, upload } = setup();

	upload.syncFileMentions(`Look at ${link(SHOT)}and again ${link(SHOT)}`);

	const first = upload.handleFileMention(SHOT);
	const second = upload.handleFileMention(SHOT);

	// sending waits for the file while it is read
	expect(upload.isReadingMention).toBe(true);

	finish(new File([new Uint8Array([1, 2, 3])], 'shot.png', { type: 'image/png' }));
	await Promise.all([first, second]);

	expect(read).toHaveBeenCalledTimes(1);
	expect(upload.isReadingMention).toBe(false);
	await expect.poll(() => upload.uploadedFiles.length).toBe(1);
	cleanup();
});

it("drops a mention's attachment though another path starts the same way", async () => {
	vi.spyOn(ReadMediaService, 'readFile').mockImplementation(
		async (path) => new File([new Uint8Array([1])], path.split('/').pop()!, { type: 'image/png' })
	);
	const { cleanup, upload } = setup();

	upload.syncFileMentions(`${link(SHOT)}${link(SHOT_BACKUP)}`);
	await upload.handleFileMention(SHOT);
	await upload.handleFileMention(SHOT_BACKUP);
	await expect.poll(() => upload.uploadedFiles.length).toBe(2);

	upload.syncFileMentions(link(SHOT_BACKUP));

	expect(upload.uploadedFiles.map((file) => file.mentionPath)).toEqual([SHOT_BACKUP]);
	cleanup();
});
