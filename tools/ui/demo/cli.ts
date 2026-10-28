/**
 * Records demo videos for pull requests and docs.
 *
 *   node demo/cli.ts pr <number>   record the ```ts demo block from a PR description
 *   node demo/cli.ts docs          record every docs/*.demo.ts script in DEMO_STORAGE
 *   node demo/cli.ts traces        write the built-in traces, and push them to DEMO_TRACES_DATASET
 *
 * Settings come from tools/ui/.env (see .env.example).
 */

import { DEMO_TRACES, generateTrace } from './fixtures/traces.ts';
import { type DemoStorage, getDemoStorage, getPublicUrl } from './storage.ts';
import { execFileSync } from 'node:child_process';
import {
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	rmSync,
	symlinkSync,
	writeFileSync
} from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { basename, join, relative, resolve } from 'node:path';

const DEMO_DIR = import.meta.dirname;
const UI_DIR = resolve(DEMO_DIR, '..');
const RUNS_DIR = join(DEMO_DIR, '.runs');
const VIDEOS_DIR = join(DEMO_DIR, 'videos');
const TRACES_DIR = join(DEMO_DIR, 'traces');
const DEMO_BLOCK = /```(?:ts|typescript)[ \t]+demo[ \t]*\r?\n([\s\S]*?)```/;
const HELPERS_IMPORT = /^import\s[^;]*from\s+['"][^'"]*helpers(?:\.ts)?['"];?\s*$/gm;
const SECTION_START = '<!-- demo:start -->';
const SECTION_END = '<!-- demo:end -->';
const envFile = join(UI_DIR, '.env');

if (existsSync(envFile)) process.loadEnvFile(envFile);

const REPO = process.env.DEMO_GITHUB_REPO ?? 'ggml-org/llama.cpp';

function run(
	cmd: string,
	args: string[],
	opts: { cwd?: string; env?: NodeJS.ProcessEnv } = {}
): string {
	return execFileSync(cmd, args, {
		cwd: opts.cwd ?? UI_DIR,
		encoding: 'utf8',
		env: opts.env
	}).trim();
}

/** Writes a demo script into .runs/, importing the harness helpers. */
function writeDemoScript(name: string, source: string): string {
	const file = join(RUNS_DIR, `${name}.demo.ts`);
	const helpers = relative(RUNS_DIR, join(DEMO_DIR, 'helpers.ts'));
	const body = source.replace(HELPERS_IMPORT, '').trim();

	mkdirSync(RUNS_DIR, { recursive: true });
	writeFileSync(file, `import { expect, test } from './${helpers}';\n\n${body}\n`);

	return file;
}

async function freePort(): Promise<number> {
	return new Promise((done, fail) => {
		const server = createServer();

		server.once('error', fail);
		server.listen(0, () => {
			const { port } = server.address() as { port: number };

			server.close(() => done(port));
		});
	});
}

/** Records the given scripts; returns the mp4 path of each one. */
async function record(scripts: string[], uiDir = UI_DIR): Promise<string[]> {
	const port = await freePort();

	execFileSync(
		'npx',
		[
			'playwright',
			'test',
			'--config=demo/playwright.config.ts',
			...scripts.map((s) => relative(UI_DIR, s))
		],
		{
			cwd: UI_DIR,
			env: { ...process.env, DEMO_UI_DIR: uiDir, DEMO_UI_PORT: String(port) },
			stdio: 'inherit'
		}
	);

	return scripts.map((s) => join(VIDEOS_DIR, `${basename(s, '.demo.ts')}.mp4`));
}

/** Animated GIF preview that GitHub renders inline. */
function makePreview(mp4: string): string {
	const gif = mp4.replace(/\.mp4$/, '.gif');
	const width = process.env.DEMO_PREVIEW_WIDTH ?? '800';

	execFileSync('ffmpeg', [
		'-y',
		'-loglevel',
		'error',
		'-i',
		mp4,
		'-vf',
		`fps=10,scale=${width}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle`,
		'-loop',
		'0',
		gif
	]);

	return gif;
}

function upload(storage: DemoStorage, files: Record<string, string>): void {
	for (const [key, file] of Object.entries(files)) {
		storage.upload(file, key);
		console.log(`uploaded ${key}`);
	}
}

function demoSection(gifUrl: string, mp4Url: string, sha: string): string {
	return [
		SECTION_START,
		`[![Demo](${gifUrl})](${mp4Url})`,
		'',
		`Recorded at ${sha.slice(0, 10)}; click the preview for the full video.`,
		SECTION_END
	].join('\n');
}

/** Replaces the managed demo section, or adds a `## Demo` section before the details. */
function withDemoSection(body: string, section: string): string {
	const start = body.indexOf(SECTION_START);
	const end = body.indexOf(SECTION_END);

	if (start !== -1 && end > start) {
		return body.slice(0, start) + section + body.slice(end + SECTION_END.length);
	}

	const block = `## Demo\n\n${section}\n\n`;
	const anchor = body.search(/^## (Additional information|Requirements)\b/m);

	return anchor === -1
		? `${body.trimEnd()}\n\n${block}`
		: body.slice(0, anchor) + block + body.slice(anchor);
}

async function recordPr(number: string): Promise<void> {
	const pr = JSON.parse(
		run('gh', ['pr', 'view', number, '--repo', REPO, '--json', 'body,headRefOid'])
	) as { body: string; headRefOid: string };
	const source = DEMO_BLOCK.exec(pr.body)?.[1];

	if (!source) throw new Error(`PR #${number} has no \`\`\`ts demo block in its description`);

	const sha = pr.headRefOid;
	const worktree = join(tmpdir(), `llama-ui-demo-pr-${number}`);

	run('git', ['fetch', '--no-tags', `https://github.com/${REPO}.git`, `pull/${number}/head`]);

	if (run('git', ['rev-parse', 'FETCH_HEAD']) !== sha) {
		throw new Error(`PR #${number} head moved while fetching, run again`);
	}

	rmSync(worktree, { force: true, recursive: true });
	run('git', ['worktree', 'prune']);
	run('git', ['worktree', 'add', '--detach', worktree, sha]);
	symlinkSync(join(UI_DIR, 'node_modules'), join(worktree, 'tools/ui/node_modules'));

	const name = `pr-${number}`;
	const script = writeDemoScript(name, source);

	try {
		const [mp4] = await record([script], join(worktree, 'tools/ui'));
		const gif = makePreview(mp4);
		const storage = getDemoStorage();

		console.log(`\nvideo:   ${mp4}\npreview: ${gif}`);

		if (!storage) {
			console.log('DEMO_STORAGE is not set: drag the mp4 into the PR description.');

			return;
		}

		const prefix = `videos/pr/${number}/${sha.slice(0, 10)}`;

		upload(storage, {
			[`${prefix}.demo.ts`]: script,
			[`${prefix}.gif`]: gif,
			[`${prefix}.mp4`]: mp4
		});

		const gifUrl = getPublicUrl(`${prefix}.gif`);
		const mp4Url = getPublicUrl(`${prefix}.mp4`);

		if (!gifUrl || !mp4Url) {
			console.log('DEMO_STORAGE_PUBLIC_URL is not set: not linking the video in the PR.');

			return;
		}

		execFileSync('gh', ['pr', 'edit', number, '--repo', REPO, '--body-file', '-'], {
			cwd: UI_DIR,
			input: withDemoSection(pr.body, demoSection(gifUrl, mp4Url, sha)),
			stdio: ['pipe', 'inherit', 'inherit']
		});
		console.log(`linked the demo in https://github.com/${REPO}/pull/${number}`);
	} finally {
		rmSync(script, { force: true });
		run('git', ['worktree', 'remove', '--force', worktree]);
	}
}

async function recordDocs(): Promise<void> {
	const storage = getDemoStorage();

	if (!storage) throw new Error('DEMO_STORAGE is not set: docs demo scripts are read from it');

	const downloaded = join(RUNS_DIR, 'docs-source');

	rmSync(downloaded, { force: true, recursive: true });

	if (!storage.downloadDir('scripts/docs', downloaded)) {
		throw new Error('no scripts/docs/*.demo.ts in DEMO_STORAGE');
	}

	const scripts = readdirSync(downloaded)
		.filter((f) => f.endsWith('.demo.ts'))
		.map((f) =>
			writeDemoScript(basename(f, '.demo.ts'), readFileSync(join(downloaded, f), 'utf8'))
		);

	try {
		const videos = await record(scripts);

		for (const mp4 of videos) {
			const name = basename(mp4, '.mp4');

			upload(storage, {
				[`videos/docs/${name}.gif`]: makePreview(mp4),
				[`videos/docs/${name}.mp4`]: mp4
			});
		}
	} finally {
		rmSync(RUNS_DIR, { force: true, recursive: true });
	}
}

/** Writes the built-in traces to demo/traces/ and uploads them to the traces dataset. */
function writeTraces(): void {
	rmSync(TRACES_DIR, { force: true, recursive: true });
	mkdirSync(TRACES_DIR, { recursive: true });

	for (const name of Object.keys(DEMO_TRACES)) {
		writeFileSync(join(TRACES_DIR, `${name}.jsonl`), `${generateTrace(name)}\n`);
		console.log(`wrote demo/traces/${name}.jsonl`);
	}

	const dataset = process.env.DEMO_TRACES_DATASET;

	if (!dataset) {
		console.log('DEMO_TRACES_DATASET is not set: not uploading.');

		return;
	}

	execFileSync(
		'hf',
		[
			'upload',
			dataset,
			TRACES_DIR,
			'.',
			'--repo-type',
			'dataset',
			'--commit-message',
			'Update demo traces'
		],
		{ stdio: 'inherit' }
	);
}

const [command, arg] = process.argv.slice(2);

if (command === 'pr' && arg) {
	await recordPr(arg);
} else if (command === 'docs') {
	await recordDocs();
} else if (command === 'traces') {
	writeTraces();
} else {
	console.error('usage: node demo/cli.ts pr <number> | docs | traces');
	process.exit(1);
}
