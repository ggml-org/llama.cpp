/**
 * Preview Spaces: puts the Llama Chat build of a branch on a static Hugging Face Space.
 *
 *   node scripts/space.ts            build this branch and deploy it, creating its Space if needed
 *   node scripts/space.ts status     print the Space of this branch, if it has one
 *   node scripts/space.ts rm         delete the Space of this branch
 *   node scripts/space.ts prune      delete preview Spaces whose branch is gone, or whose pull request is closed
 *
 * It uses the `hf` CLI and its login, or HF_TOKEN. Settings come from ui/apps/chat/.env (see .env.example).
 */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, cpSync, existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createInterface } from 'node:readline/promises';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const UI_DIR = resolve(import.meta.dirname, '..');
const DIST_DIR = join(UI_DIR, 'dist');
const HF_ENDPOINT = 'https://huggingface.co';
const SPACE_PREFIX = 'llama-ui-';
const SPACE_NAME_MAX_LENGTH = 96;
const BRANCH_HASH_LENGTH = 8;
const DEFAULT_SPACE_TITLE = 'Llama Chat preview';
const SPACE_ID_REGEX = /^[\w.-]+\/[\w.-]+$/;
// marks the Spaces this script made: it never deploys to or deletes any other
const PREVIEW_TAG = 'llama-ui-preview';
// names the repository and branch a Space was built from, so the upstream repository can find it by tag
const SOURCE_TAG_PREFIX = 'llama-ui-src-';
const SOURCE_TAG_HASH_LENGTH = 12;
// the card's machine-readable line: what the Space was built from
const SOURCE_COMMENT_REGEX = /<!-- llama-ui-preview-source (\{.*?\}) -->/;
// a preview has no llama-server of its own: it talks to the one on the reviewer's computer
const DEFAULT_PREVIEW_SERVER = 'http://localhost:9931';
const DEFAULT_UPSTREAM = 'ggml-org/llama.cpp';
const GITHUB_API = 'https://api.github.com';
const VISIBILITIES = ['public', 'protected', 'private'];
const COMMANDS = ['deploy', 'status', 'rm', 'prune'];

export const USAGE = `Usage: npm run space -- [deploy|status|rm|prune] [options]

  deploy (default)   build this branch and deploy it, creating its Space if needed
  status             print the Space of this branch, if it has one
  rm                 delete the Space of this branch
  prune              delete preview Spaces whose branch is gone from the remote, or whose
                     pull request to the upstream repository is closed or merged

Options:
  --branch <name>      the branch, instead of the one checked out
  --owner <name>       the Hugging Face user or organization (default: UI_PREVIEW_SPACE_OWNER, or the login)
  --space <owner/name> a Space of that name, instead of the one named after the branch
  --title <title>      the title on the Space's card (default: Llama Chat preview)
  --server <url>       the llama-server the preview talks to, in its config.json
                       (default: ${DEFAULT_PREVIEW_SERVER})
  --upstream <repo>    the repository prune reads pull requests from (default: ${DEFAULT_UPSTREAM})
  --visibility <v>     ${VISIBILITIES.join(', ')}; used when the Space is created (default: public)
  --if-exists          deploy only if the branch already has a Space
  --wip                deploy the working tree as it is, even if it is not pushed
  --remote <name>      the git remote prune reads the branches from (default: the one this branch tracks)
  -y, --yes            prune without asking
  -h, --help           print this help`;

export interface SpaceOptions {
	branch?: string;
	command: string;
	help: boolean;
	ifExists: boolean;
	owner?: string;
	space?: string;
	server: string;
	title?: string;
	remote?: string;
	upstream: string;
	visibility: string;
	wip: boolean;
	yes: boolean;
}

interface Space {
	host?: string;
	id: string;
	tags?: string[];
}

export function parseSpaceArgs(argv: string[]): SpaceOptions {
	const { positionals, values } = parseArgs({
		allowPositionals: true,
		args: argv,
		options: {
			branch: { type: 'string' },
			help: { default: false, short: 'h', type: 'boolean' },
			'if-exists': { default: false, type: 'boolean' },
			owner: { type: 'string' },
			remote: { type: 'string' },
			server: { default: DEFAULT_PREVIEW_SERVER, type: 'string' },
			space: { type: 'string' },
			title: { type: 'string' },
			upstream: { default: DEFAULT_UPSTREAM, type: 'string' },
			visibility: { default: 'public', type: 'string' },
			wip: { default: false, type: 'boolean' },
			yes: { default: false, short: 'y', type: 'boolean' }
		}
	});
	const command = positionals[0] ?? 'deploy';

	if (positionals.length > 1 || !COMMANDS.includes(command)) {
		throw new Error(`unknown command: ${positionals.join(' ')}`);
	}

	if (values.space !== undefined && !SPACE_ID_REGEX.test(values.space)) {
		throw new Error('--space is <owner>/<name>');
	}

	if (values.server && !/^https?:\/\/[^/]/.test(values.server)) {
		throw new Error('--server is an http(s) URL');
	}

	if (!SPACE_ID_REGEX.test(values.upstream)) {
		throw new Error('--upstream is <owner>/<repo>');
	}

	if (!VISIBILITIES.includes(values.visibility)) {
		throw new Error(`--visibility is one of: ${VISIBILITIES.join(', ')}`);
	}

	return {
		branch: values.branch,
		command,
		help: values.help,
		ifExists: values['if-exists'],
		owner: values.owner,
		remote: values.remote,
		server: values.server,
		space: values.space,
		title: values.title,
		upstream: values.upstream,
		visibility: values.visibility,
		wip: values.wip,
		yes: values.yes
	};
}

/** The Space name of a branch: `<owner>/feat/voice-mode` and `feat/voice-mode` both give `llama-ui-feat-voice-mode`. */
export function spaceName(branch: string, owner: string): string {
	const own = `${owner.toLowerCase()}/`;
	const rest = branch.toLowerCase().startsWith(own) ? branch.slice(own.length) : branch;
	const slug = rest
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '-')
		.replace(/^-|-$/g, '');

	if (!slug) throw new Error(`no Space name for branch "${branch}"`);

	const name = SPACE_PREFIX + slug;

	if (name.length <= SPACE_NAME_MAX_LENGTH) return name;

	// too long: cut it, and keep it unique with a hash of the branch
	const hash = createHash('sha1').update(branch).digest('hex').slice(0, BRANCH_HASH_LENGTH);
	const head = name.slice(0, SPACE_NAME_MAX_LENGTH - hash.length - 1).replace(/-$/, '');

	return `${head}-${hash}`;
}

/** What a Space was built from: the GitHub repository (`owner/name`, '' when unknown), the branch and the commit. */
export interface SpaceSource {
	branch: string;
	repo: string;
	sha: string;
}

/**
 * The tag that names the repository and branch a Space was built from. The upstream repository
 * finds the preview of a pull request by it, without knowing the Hugging Face account it is on.
 */
export function sourceTag(repo: string, branch: string): string {
	const hash = createHash('sha1').update(`${repo.toLowerCase()}:${branch}`).digest('hex');

	return SOURCE_TAG_PREFIX + hash.slice(0, SOURCE_TAG_HASH_LENGTH);
}

/** The README of the Space. Its headers are the ones llama-server serves the UI with (cross-origin isolation). */
export function spaceCard(info: SpaceSource & { server?: string; title?: string }): string {
	const source = info.repo ? `[${info.repo}](https://github.com/${info.repo})` : 'llama.cpp';
	const tags = [PREVIEW_TAG, ...(info.repo ? [sourceTag(info.repo, info.branch)] : [])];
	const server = info.server
		? `It has no server of its own: it talks to the llama-server on ${info.server} of whoever opens it.`
		: 'It has no server of its own.';
	const meta = JSON.stringify({ branch: info.branch, repo: info.repo, sha: info.sha });

	return `---
title: ${info.title ?? DEFAULT_SPACE_TITLE}
colorFrom: gray
colorTo: gray
sdk: static
pinned: false
tags:
${tags.map((tag) => `  - ${tag}`).join('\n')}
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
---

A preview of Llama Chat, built from the branch \`${info.branch}\` of ${source} at ${info.sha.slice(0, 10)}.

${server}

<!-- llama-ui-preview-source ${meta} -->
`;
}

/** What a Space was built from, read back from its card; null for a card without it. */
export function readSpaceSource(card: string): SpaceSource | null {
	const match = SOURCE_COMMENT_REGEX.exec(card);

	if (!match) return null;

	try {
		const source = JSON.parse(match[1]) as Partial<SpaceSource>;

		return typeof source.branch === 'string' && typeof source.repo === 'string'
			? { branch: source.branch, repo: source.repo, sha: String(source.sha ?? '') }
			: null;
	} catch {
		return null;
	}
}

/** Whether a branch's pull requests leave its preview wanted: open, done (all closed or merged), or none at all. */
export function pullRequestsState(pulls: { state: string }[]): 'done' | 'none' | 'open' {
	if (pulls.length === 0) return 'none';

	return pulls.some((pull) => pull.state === 'open') ? 'open' : 'done';
}

function run(cmd: string, args: string[]): string {
	return execFileSync(cmd, args, { cwd: UI_DIR, encoding: 'utf8' }).trim();
}

function tryRun(cmd: string, args: string[]): string | null {
	try {
		return execFileSync(cmd, args, {
			cwd: UI_DIR,
			encoding: 'utf8',
			stdio: ['ignore', 'pipe', 'ignore']
		}).trim();
	} catch {
		return null;
	}
}

function exec(cmd: string, args: string[]): void {
	execFileSync(cmd, args, { cwd: UI_DIR, stdio: 'inherit' });
}

let token: string | undefined;

function hfToken(): string {
	token ??= process.env.HF_TOKEN || tryRun('hf', ['auth', 'token']) || '';

	if (!token)
		throw new Error('not logged in to Hugging Face: run `hf auth login`, or set HF_TOKEN');

	return token;
}

/** A GET on the Hub API; null when there is no such thing. */
async function hub<T>(path: string): Promise<T | null> {
	const response = await fetch(`${HF_ENDPOINT}/api/${path}`, {
		headers: { Authorization: `Bearer ${hfToken()}` }
	});

	if (response.status === 404) return null;

	if (!response.ok) throw new Error(`Hugging Face answered ${response.status} for ${path}`);

	return (await response.json()) as T;
}

async function spaceOwner(opts: SpaceOptions): Promise<string> {
	const owner = opts.owner ?? process.env.UI_PREVIEW_SPACE_OWNER;

	if (owner) return owner;

	const me = await hub<{ name: string }>('whoami-v2');

	if (!me) throw new Error('could not read the Hugging Face login');

	return me.name;
}

function currentBranch(opts: SpaceOptions): string {
	const branch =
		opts.branch ?? process.env.GITHUB_REF_NAME ?? run('git', ['branch', '--show-current']);

	if (!branch) throw new Error('no branch is checked out: pass --branch <name>');

	return branch;
}

/** The branch the command is about, and the id of its Space. */
async function branchSpace(opts: SpaceOptions): Promise<{ branch: string; id: string }> {
	const branch = currentBranch(opts);

	if (opts.space) return { branch, id: opts.space };

	const owner = await spaceOwner(opts);

	return { branch, id: `${owner}/${spaceName(branch, owner)}` };
}

async function getSpace(id: string): Promise<Space | null> {
	const space = await hub<Space>(`spaces/${id}`);

	if (space && !space.tags?.includes(PREVIEW_TAG)) {
		throw new Error(`${id} exists and is not a preview Space: it was not made by this script`);
	}

	return space;
}

/** The `owner/name` of the GitHub repository behind a remote, or '' when it is not on GitHub. */
function githubRepo(remote: string | null): string {
	const url = remote && tryRun('git', ['remote', 'get-url', remote]);

	return (url && /github\.com[:/](.+?)(?:\.git)?$/.exec(url)?.[1]) || '';
}

// read from the branch config, not @{upstream}: a remote may not fetch the branch it is pushed to
function tracked(key: 'merge' | 'remote'): string | null {
	const branch = tryRun('git', ['branch', '--show-current']);

	return (branch && tryRun('git', ['config', `branch.${branch}.${key}`])) || null;
}

function trackedRemote(): string | null {
	return tracked('remote');
}

/** A Space shows what is on the remote: refuses a tree with changes to the UI, or a HEAD that is not pushed. */
function assertPushed(): void {
	if (run('git', ['status', '--porcelain', '--', resolve(UI_DIR, '../..')])) {
		throw new Error('the UI has uncommitted changes: commit them, or pass --wip');
	}

	const remote = trackedRemote();
	const ref = tracked('merge');
	const pushed = remote && ref && tryRun('git', ['ls-remote', remote, ref])?.split('\t')[0];

	if (run('git', ['rev-parse', 'HEAD']) !== pushed) {
		throw new Error('this branch is not pushed as it is here: push it, or pass --wip');
	}
}

async function deploy(opts: SpaceOptions): Promise<void> {
	const { branch, id } = await branchSpace(opts);
	const existing = await getSpace(id);

	if (!existing && opts.ifExists) {
		console.log(`${branch} has no Space (${id}): nothing to deploy`);

		return;
	}

	if (!process.env.CI && !opts.wip) assertPushed();

	exec('npm', ['run', 'build']);

	// staged in a copy, so the card does not end up in dist, which llama-server embeds
	const staged = mkdtempSync(join(tmpdir(), 'llama-ui-space-'));
	const sha = run('git', ['rev-parse', 'HEAD']);
	const repo = process.env.GITHUB_REPOSITORY ?? githubRepo(trackedRemote());

	try {
		cpSync(DIST_DIR, staged, { recursive: true });
		writeFileSync(
			join(staged, 'README.md'),
			spaceCard({ branch, repo, server: opts.server, sha, title: opts.title })
		);
		// the app reads config.json on start: the build ships it empty, for its own origin
		writeFileSync(join(staged, 'config.json'), `${JSON.stringify({ server: opts.server })}\n`);

		if (!existing) {
			exec('hf', [
				'repos',
				'create',
				id,
				'--repo-type',
				'space',
				'--space-sdk',
				'static',
				`--${opts.visibility}`
			]);
		}

		exec('hf', [
			'upload',
			id,
			staged,
			'.',
			'--repo-type',
			'space',
			'--delete',
			'*',
			'--commit-message',
			`${branch} at ${sha.slice(0, 10)}${opts.wip ? ' (work in progress)' : ''}`
		]);
	} finally {
		rmSync(staged, { force: true, recursive: true });
	}

	console.log(`\n${branch} is at ${(await getSpace(id))?.host}`);
}

async function status(opts: SpaceOptions): Promise<void> {
	const { branch, id } = await branchSpace(opts);
	const space = await getSpace(id);

	console.log(space ? `${branch} is at ${space.host}` : `${branch} has no Space (${id})`);

	// a workflow reads this to skip the build of a branch without a Space
	if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `exists=${!!space}\n`);
}

function deleteSpace(id: string): void {
	exec('hf', ['repos', 'delete', id, '--repo-type', 'space', '--yes']);
}

async function remove(opts: SpaceOptions): Promise<void> {
	const { branch, id } = await branchSpace(opts);

	if (!(await getSpace(id))) {
		console.log(`${branch} has no Space (${id})`);

		return;
	}

	deleteSpace(id);
	console.log(`deleted ${id}`);
}

/** The card of a Space, as text; '' when it cannot be read. */
async function spaceReadme(id: string): Promise<string> {
	const response = await fetch(`${HF_ENDPOINT}/spaces/${id}/raw/main/README.md`, {
		headers: { Authorization: `Bearer ${hfToken()}` }
	});

	return response.ok ? response.text() : '';
}

/** The pull requests from a branch of `repo` to `upstream`, open or closed; null when GitHub does not answer. */
async function pullRequests(
	upstream: string,
	source: SpaceSource
): Promise<{ state: string }[] | null> {
	const auth = process.env.GITHUB_TOKEN || tryRun('gh', ['auth', 'token']);
	const head = `${source.repo.split('/')[0]}:${source.branch}`;
	const response = await fetch(
		`${GITHUB_API}/repos/${upstream}/pulls?state=all&per_page=100&head=${encodeURIComponent(head)}`,
		{
			headers: {
				Accept: 'application/vnd.github+json',
				...(auth ? { Authorization: `Bearer ${auth}` } : {})
			}
		}
	);

	return response.ok ? ((await response.json()) as { state: string }[]) : null;
}

/** Why a preview Space is no longer wanted, or null to keep it. */
async function staleReason(
	space: Space,
	branches: Set<string> | null,
	upstream: string
): Promise<string | null> {
	if (branches && !branches.has(space.id)) return 'its branch is gone';

	const source = readSpaceSource(await spaceReadme(space.id));

	if (!source?.repo) return null;

	const pulls = await pullRequests(upstream, source);

	return pulls && pullRequestsState(pulls) === 'done'
		? `its pull request to ${upstream} is closed`
		: null;
}

async function prune(opts: SpaceOptions): Promise<void> {
	const owner = await spaceOwner(opts);
	const remote = opts.remote ?? trackedRemote();
	// without a remote, only the pull requests decide
	const branches = remote
		? new Set(
				run('git', ['ls-remote', '--heads', remote])
					.split('\n')
					.filter(Boolean)
					.map((line) => `${owner}/${spaceName(line.split('refs/heads/')[1], owner)}`)
			)
		: null;
	const spaces = await hub<Space[]>(
		`spaces?author=${encodeURIComponent(owner)}&filter=${PREVIEW_TAG}&full=true&limit=1000`
	);
	const stale: { id: string; reason: string }[] = [];

	for (const space of spaces ?? []) {
		if (!space.tags?.includes(PREVIEW_TAG)) continue;

		const reason = await staleReason(space, branches, opts.upstream);

		if (reason) stale.push({ id: space.id, reason });
	}

	if (stale.length === 0) {
		console.log(`every preview Space of ${owner} is still wanted`);

		return;
	}

	console.log(
		`No longer wanted:\n${stale.map(({ id, reason }) => `  ${id}: ${reason}`).join('\n')}`
	);

	if (!opts.yes) {
		const prompt = createInterface({ input: process.stdin, output: process.stdout });
		const answer = await prompt.question(`Delete ${stale.length}? [y/N] `);

		prompt.close();

		if (answer.trim().toLowerCase() !== 'y') return;
	}

	for (const { id } of stale) deleteSpace(id);
}

export async function main(argv: string[]): Promise<void> {
	const envFile = join(UI_DIR, '.env');

	if (existsSync(envFile)) process.loadEnvFile(envFile);

	const opts = parseSpaceArgs(argv);

	if (opts.help) {
		console.log(USAGE);

		return;
	}

	await { deploy, prune, rm: remove, status }[opts.command]!(opts);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
	main(process.argv.slice(2)).catch((error: Error) => {
		console.error(`space: ${error.message}`);
		process.exit(1);
	});
}
