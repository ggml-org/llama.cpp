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
const UI_ROOT = resolve(UI_DIR, '../..');
const DIST_DIR = join(UI_DIR, 'dist');
const HF_ENDPOINT = 'https://huggingface.co';
const SPACE_PREFIX = 'llama-ui-';
const SPACE_NAME_MAX_LENGTH = 96;
const BRANCH_HASH_LENGTH = 8;
const SHORT_SHA_LENGTH = 10;
const DEFAULT_SPACE_TITLE = 'Llama Chat preview';
const SPACE_ID_REGEX = /^[\w.-]+\/[\w.-]+$/;
const SERVER_URL_REGEX = /^https?:\/\/[^/]/;
// marks the Spaces this script made: it never deploys to or deletes any other
const PREVIEW_TAG = 'llama-ui-preview';
// the card's machine-readable line: what the Space was built from
const SOURCE_COMMENT_REGEX = /<!-- llama-ui-preview-source (\{.*?\}) -->/;
// a preview has no llama-server of its own: it talks to the one on the reviewer's computer
const DEFAULT_PREVIEW_SERVER = 'http://localhost:9931';
const DEFAULT_UPSTREAM = 'ggml-org/llama.cpp';
const GITHUB_API = 'https://api.github.com';
// the GitHub environment a deploy is recorded in: ui-preview-space-link.yml reads it by this name
const DEPLOYMENT_ENVIRONMENT = 'Llama Chat preview';
const GITHUB_REMOTE_REGEX = /github\.com[:/](.+?)(?:\.git)?$/;
const BRANCH_REF_PREFIX = 'refs/heads/';
// `branch.<name>.remote` of a branch that tracks a local one
const LOCAL_REMOTE = '.';
// not handed to the build, which runs the branch's code and its dependencies
const TOKEN_VARIABLES = ['HF_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'];
const DEFAULT_VISIBILITY = 'public';
const VISIBILITIES = [DEFAULT_VISIBILITY, 'protected', 'private'];
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
  --title <title>      the title on the Space's card (default: ${DEFAULT_SPACE_TITLE})
  --server <url>       the llama-server the preview talks to, in its config.json
                       (default: ${DEFAULT_PREVIEW_SERVER})
  --upstream <repo>    the repository prune reads pull requests from (default: ${DEFAULT_UPSTREAM})
  --visibility <v>     ${VISIBILITIES.join(', ')}; used when the Space is created (default: ${DEFAULT_VISIBILITY})
  --if-exists          deploy only if the branch already has a Space
  --built              deploy dist as it is, without building
  --wip                deploy the working tree as it is, even if it is not pushed
  --remote <name>      the git remote the branch is on (default: the one it is pushed to or tracks)
  -y, --yes            prune without asking
  -h, --help           print this help`;

export interface SpaceOptions {
	branch?: string;
	built: boolean;
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
	lastModified?: string;
	private?: boolean;
	tags?: string[];
}

export interface PullRequest {
	closed_at: string | null;
	state: string;
}

export function parseSpaceArgs(argv: string[]): SpaceOptions {
	const { positionals, values } = parseArgs({
		allowPositionals: true,
		args: argv,
		options: {
			branch: { type: 'string' },
			built: { default: false, type: 'boolean' },
			help: { default: false, short: 'h', type: 'boolean' },
			'if-exists': { default: false, type: 'boolean' },
			owner: { type: 'string' },
			remote: { type: 'string' },
			server: { default: DEFAULT_PREVIEW_SERVER, type: 'string' },
			space: { type: 'string' },
			title: { type: 'string' },
			upstream: { default: DEFAULT_UPSTREAM, type: 'string' },
			visibility: { default: DEFAULT_VISIBILITY, type: 'string' },
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

	if (!SERVER_URL_REGEX.test(values.server)) {
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
		built: values.built,
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

function spaceId(branch: string, owner: string): string {
	return `${owner}/${spaceName(branch, owner)}`;
}

/** What a Space was built from: the GitHub repository (`owner/name`, '' when unknown), the branch and the commit. */
export interface SpaceSource {
	branch: string;
	repo: string;
	sha: string;
}

/**
 * Whether a Space built from `built` is the Space of `branch` in this checkout, whose remotes are `repos`.
 * A repository that is not known matches any.
 */
export function isSpaceOf(
	built: Pick<SpaceSource, 'branch' | 'repo'>,
	branch: string,
	repos: string[]
): boolean {
	const known = repos.map((repo) => repo.toLowerCase());

	return (
		built.branch === branch &&
		(!built.repo || known.length === 0 || known.includes(built.repo.toLowerCase()))
	);
}

/**
 * Whether prune may judge a Space: only one built from this repository and named after its branch.
 * A Space of another name was named by hand (--space), and is never pruned.
 */
export function isPrunable(
	id: string,
	owner: string,
	source: SpaceSource | null,
	repo: string
): source is SpaceSource {
	return (
		!!source &&
		source.repo.toLowerCase() === repo.toLowerCase() &&
		id.toLowerCase() === spaceId(source.branch, owner).toLowerCase()
	);
}

/** The README of the Space. Its headers are the ones llama-server serves the UI with (cross-origin isolation). */
export function spaceCard(info: SpaceSource & { server?: string; title?: string }): string {
	const source = info.repo ? `[${info.repo}](https://github.com/${info.repo})` : 'llama.cpp';
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
  - ${PREVIEW_TAG}
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
---

A preview of Llama Chat, built from the branch \`${info.branch}\` of ${source} at ${info.sha.slice(0, SHORT_SHA_LENGTH)}.

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

/**
 * Whether a branch's pull requests are over for its preview: none is open, and one closed after
 * the Space was last deployed. A deploy after the last one closed means the branch is in use again.
 */
export function pullRequestsDone(pulls: PullRequest[], deployedAt: string | undefined): boolean {
	if (!deployedAt || pulls.some((pull) => pull.state === 'open')) return false;

	return pulls.some(
		(pull) => pull.closed_at && Date.parse(pull.closed_at) > Date.parse(deployedAt)
	);
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

function exec(cmd: string, args: string[], env = process.env): void {
	execFileSync(cmd, args, { cwd: UI_DIR, env, stdio: 'inherit' });
}

function withoutTokens(): NodeJS.ProcessEnv {
	return Object.fromEntries(
		Object.entries(process.env).filter(([name]) => !TOKEN_VARIABLES.includes(name))
	);
}

let token: string | undefined;

function hfHeaders(): { Authorization: string } {
	token ??= process.env.HF_TOKEN || tryRun('hf', ['auth', 'token']) || '';

	if (!token)
		throw new Error('not logged in to Hugging Face: run `hf auth login`, or set HF_TOKEN');

	return { Authorization: `Bearer ${token}` };
}

/** A GET on the Hub API; null when there is no such thing. */
async function hub<T>(path: string): Promise<T | null> {
	const response = await fetch(`${HF_ENDPOINT}/api/${path}`, { headers: hfHeaders() });

	if (response.status === 404) return null;

	if (!response.ok) throw new Error(`Hugging Face answered ${response.status} for ${path}`);

	return (await response.json()) as T;
}

/** A call on the GitHub API of a repository: a POST when it has a body. */
function github(path: string, body?: object): Promise<Response> {
	const auth = process.env.GITHUB_TOKEN || tryRun('gh', ['auth', 'token']);

	return fetch(`${GITHUB_API}/repos/${path}`, {
		body: body && JSON.stringify(body),
		headers: {
			Accept: 'application/vnd.github+json',
			...(auth ? { Authorization: `Bearer ${auth}` } : {})
		},
		method: body ? 'POST' : 'GET'
	});
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

async function getSpace(id: string): Promise<Space | null> {
	const space = await hub<Space>(`spaces/${id}`);

	if (space && !space.tags?.includes(PREVIEW_TAG)) {
		throw new Error(`${id} exists and is not a preview Space: it was not made by this script`);
	}

	return space;
}

function spaceUrl(id: string, space: Space | null): string {
	return space?.host ?? `${HF_ENDPOINT}/spaces/${id}`;
}

/** The card of a Space, as text; '' when it cannot be read. */
async function spaceReadme(id: string): Promise<string> {
	const response = await fetch(`${HF_ENDPOINT}/spaces/${id}/raw/main/README.md`, {
		headers: hfHeaders()
	});

	return response.ok ? response.text() : '';
}

/**
 * The remotes the branch may be on: the one named, else the ones git pushes the checked out branch to or tracks.
 * Read from the config, not @{upstream}: a remote may not fetch the branch it is pushed to.
 */
function branchRemotes(opts: SpaceOptions): string[] {
	if (opts.remote) return [opts.remote];

	const config = (key: string) => tryRun('git', ['config', key]);
	const local = tryRun('git', ['branch', '--show-current']);
	const remotes = [
		local && config(`branch.${local}.pushRemote`),
		local && config(`branch.${local}.remote`),
		config('remote.pushDefault')
	].filter((remote): remote is string => !!remote && remote !== LOCAL_REMOTE);

	return [...new Set(remotes)];
}

/** The GitHub repository (`owner/name`) behind a remote, or '' when it is not on GitHub. */
function sourceRepo(remote: string | undefined): string {
	if (process.env.GITHUB_REPOSITORY) return process.env.GITHUB_REPOSITORY;

	const url = remote && tryRun('git', ['remote', 'get-url', remote]);

	return (url && GITHUB_REMOTE_REGEX.exec(url)?.[1]) || '';
}

/** The GitHub repositories a branch of this checkout can be in: the one named or of the workflow, else every remote. */
function checkoutRepos(opts: SpaceOptions): string[] {
	const remotes =
		opts.remote || process.env.GITHUB_REPOSITORY
			? [opts.remote]
			: (tryRun('git', ['remote'])?.split('\n') ?? []);

	return remotes.map(sourceRepo).filter(Boolean);
}

/**
 * A Space shows what is on the remote: refuses a tree with changes to the UI, and a HEAD that is not
 * the branch as it is pushed. Gives the remote that has it.
 */
function pushedRemote(branch: string, remotes: string[]): string {
	if (run('git', ['status', '--porcelain', '--', UI_ROOT])) {
		throw new Error('the UI has uncommitted changes: commit them, or pass --wip');
	}

	const head = run('git', ['rev-parse', 'HEAD']);
	const remote = remotes.find(
		(name) =>
			tryRun('git', ['ls-remote', name, BRANCH_REF_PREFIX + branch])?.split('\t')[0] === head
	);

	if (!remote) {
		throw new Error(
			`what is checked out is not ${branch} as it is pushed: push it, pass --remote <name>, or pass --wip`
		);
	}

	return remote;
}

interface Target {
	branch: string;
	id: string;
	/** null when the branch has no Space, also when another branch's Space holds its name */
	space: Space | null;
	/** the branch whose Space holds the name */
	takenBy?: string;
}

/** The branch the command is about, and its Space. Two branches can get the same Space name: the Space is of the one it was built from. */
async function target(opts: SpaceOptions): Promise<Target> {
	const branch = currentBranch(opts);

	if (opts.space) return { branch, id: opts.space, space: await getSpace(opts.space) };

	const id = spaceId(branch, await spaceOwner(opts));
	const space = await getSpace(id);
	const built = space && readSpaceSource(await spaceReadme(id));

	if (!space || (built && isSpaceOf(built, branch, checkoutRepos(opts)))) {
		return { branch, id, space };
	}

	const takenBy = !built
		? 'an unknown branch'
		: built.repo
			? `${built.branch} of ${built.repo}`
			: built.branch;

	return { branch, id, space: null, takenBy };
}

/**
 * Records the deploy on the GitHub repository it was built from. Only who can push there can do that,
 * so the upstream repository takes it as proof of which Space is the preview of a pull request.
 */
async function recordDeployment(source: SpaceSource, id: string, url: string): Promise<void> {
	try {
		const created = await github(`${source.repo}/deployments`, {
			auto_merge: false,
			description: `${id} at ${source.sha.slice(0, SHORT_SHA_LENGTH)}`,
			environment: DEPLOYMENT_ENVIRONMENT,
			payload: { branch: source.branch, space: id },
			ref: source.sha,
			required_contexts: []
		});

		if (!created.ok) throw new Error(`GitHub answered ${created.status}`);

		const deployment = (await created.json()) as { id: number };

		await github(`${source.repo}/deployments/${deployment.id}/statuses`, {
			environment_url: url,
			state: 'success'
		});
	} catch (error) {
		console.warn(
			`the deploy is not recorded on ${source.repo} (${(error as Error).message}): a pull request gets no link to it`
		);
	}
}

async function deploy(opts: SpaceOptions): Promise<void> {
	const { branch, id, space, takenBy } = await target(opts);

	if (takenBy) {
		throw new Error(`${id} is the Space of ${takenBy}: pass --space <owner/name> for another`);
	}

	if (!space && opts.ifExists) {
		console.log(`${branch} has no Space (${id}): nothing to deploy`);

		return;
	}

	const remotes = branchRemotes(opts);
	// a workflow has just checked out the pushed branch
	const pushed = !!process.env.CI || !opts.wip;
	const remote = process.env.CI || opts.wip ? remotes[0] : pushedRemote(branch, remotes);

	if (!opts.built) exec('npm', ['run', 'build'], withoutTokens());

	// staged in a copy, so the card does not end up in dist, which llama-server embeds
	const staged = mkdtempSync(join(tmpdir(), 'llama-ui-space-'));
	const source = { branch, repo: sourceRepo(remote), sha: run('git', ['rev-parse', 'HEAD']) };
	const short = source.sha.slice(0, SHORT_SHA_LENGTH);

	try {
		cpSync(DIST_DIR, staged, { recursive: true });
		writeFileSync(
			join(staged, 'README.md'),
			spaceCard({ ...source, server: opts.server, title: opts.title })
		);
		// the app reads config.json on start: the build ships it empty, for its own origin
		writeFileSync(join(staged, 'config.json'), `${JSON.stringify({ server: opts.server })}\n`);

		if (!space) {
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
			`${branch} at ${short}${pushed ? '' : ' (work in progress)'}`
		]);
	} finally {
		rmSync(staged, { force: true, recursive: true });
	}

	const deployed = await getSpace(id);
	const url = spaceUrl(id, deployed);

	// a private Space is no use to a reviewer
	if (pushed && source.repo && !deployed?.private) await recordDeployment(source, id, url);

	console.log(`\n${branch} is at ${url}`);
}

async function status(opts: SpaceOptions): Promise<void> {
	const { branch, id, space, takenBy } = await target(opts);

	console.log(
		space
			? `${branch} is at ${spaceUrl(id, space)}`
			: `${branch} has no Space (${id}${takenBy ? ` is the Space of ${takenBy}` : ''})`
	);

	// a workflow reads this to skip the build of a branch without a Space
	if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `exists=${!!space}\n`);
}

function deleteSpace(id: string): void {
	exec('hf', ['repos', 'delete', id, '--repo-type', 'space', '--yes']);
}

async function remove(opts: SpaceOptions): Promise<void> {
	const { branch, id, space, takenBy } = await target(opts);

	if (!space) {
		console.log(`${branch} has no Space (${id}${takenBy ? ` is the Space of ${takenBy}` : ''})`);

		return;
	}

	deleteSpace(id);
	console.log(`deleted ${id}`);
}

/** The pull requests from a branch of `repo` to `upstream`, open or closed; null when GitHub does not answer. */
async function pullRequests(upstream: string, source: SpaceSource): Promise<PullRequest[] | null> {
	const head = `${source.repo.split('/')[0]}:${source.branch}`;
	const response = await github(
		`${upstream}/pulls?state=all&per_page=100&head=${encodeURIComponent(head)}`
	);

	return response.ok ? ((await response.json()) as PullRequest[]) : null;
}

/** Why a preview Space is no longer wanted, or null to keep it. */
async function staleReason(
	space: Space,
	source: SpaceSource,
	branches: Set<string> | null,
	upstream: string
): Promise<string | null> {
	if (branches && !branches.has(source.branch)) return 'its branch is gone';

	if (!source.repo) return null;

	const pulls = await pullRequests(upstream, source);

	return pulls && pullRequestsDone(pulls, space.lastModified)
		? `its pull request to ${upstream} is closed`
		: null;
}

async function prune(opts: SpaceOptions): Promise<void> {
	const owner = await spaceOwner(opts);
	const remote = branchRemotes(opts)[0];
	const repo = sourceRepo(remote);
	// without a remote, only the pull requests decide
	const branches = remote
		? new Set(
				run('git', ['ls-remote', '--heads', remote])
					.split('\n')
					.filter(Boolean)
					.map((line) => line.split(BRANCH_REF_PREFIX)[1])
			)
		: null;
	const spaces = await hub<Space[]>(
		`spaces?author=${encodeURIComponent(owner)}&filter=${PREVIEW_TAG}&full=true&limit=1000`
	);
	const stale: { id: string; reason: string }[] = [];

	for (const space of spaces ?? []) {
		if (!space.tags?.includes(PREVIEW_TAG)) continue;

		const source = readSpaceSource(await spaceReadme(space.id));

		if (!isPrunable(space.id, owner, source, repo)) continue;

		const reason = await staleReason(space, source, branches, opts.upstream);

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
