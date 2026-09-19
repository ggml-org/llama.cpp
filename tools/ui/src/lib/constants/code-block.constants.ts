// Constants for the markdown code-block renderer: language/fence handling and CSS classes.

/** Parsing and escaping helpers for the markdown code-block renderer. */
export const CODE_BLOCK = {
	AMPERSAND_REGEX: /&/g,
	/** Language fallback used when no language is specified. */
	DEFAULT_LANGUAGE: 'text',
	/** Default when file name cannot be retrieved from code block or text above */
	DOWNLOAD_NAME_DEFAULT_PREFIX: 'llama_',
	DOWNLOAD_NAME_SCAN_BYTES_BACK: 500,
	/** If a code block has no file name, scan the text above for a valid file name */
	DOWNLOAD_NAME_SCAN_LINES_BACK: 3,
	/* valid break characters before and after a file name, like here is droppath/script.sh: */
	DOWNLOAD_NAME_STOP_MARKS: ':;,!?/\\',
	/* prioritized wrapping chars around file name. like save as **script.js** and run in node.js */
	DOWNLOAD_NAME_WRAP_MARKS: '`"\'‘’*()[]{}«»',

	/** Matches opening/closing markdown code fences. */
	FENCE_PATTERN: /^```|\n```/g,

	FILE_EXT_VALID_CHAR_REGEX: /^[a-zA-Z0-9]+$/,
	// filter file name from MD code block header, like ```python title="script.py"
	FILE_NAME_HEADER_REGEX:
		/(?:^|\s)(?:[a-zA-Z0-9_-]+=)?["'`]?([^"'`\s]+\.[a-zA-Z][a-zA-Z0-9]{0,9})["'`]?(?:\s|$)/i,
	FILE_NAME_MAX_EXT_LENGTH: 10,
	FILE_NAME_MAX_LENGTH: 40,
	// allowed characters for a file name (stick to POSIX portable filenames char set)
	FILE_NAME_VALID_CHAR_REGEX: /^[a-zA-Z0-9._ -]+$/,
	GT_REGEX: />/g,
	ID_PREFIX: 'code',
	/** Matches the language specifier at the start of a code fence. */
	LANG_PATTERN: /^(\w*)\n?/,
	LT_REGEX: /</g,
	// Matches the `text:` prefix that file-type identifiers use to denote a
	// plain-text language (e.g. `text:typescript`). Used by tool-call renderers
	// to recover the underlying highlight.js language.
	TEXT_LANGUAGE_PREFIX_REGEX: /^text:/,
	// Whitespace-only empty lines (between start of string and first non-empty line).
	// Used by trimCodePadding to drop leading/trailing phantom blank rows from LLM
	// payload wrappers without touching internal blank lines.
	TRIM_LEADING_PADDING_REGEX: /^(?:[ \t]*\n)+/,
	TRIM_TRAILING_PADDING_REGEX: /(?:\n[ \t]*)+$/
} as const;

// Matches either Unix or Windows path separators so `String.split(REGEX)` can
// recover the trailing file-name segment from either `/foo/bar.txt` or
// `C:\foo\bar.txt`. Used wherever a parameter accepts a user-supplied path.
export const FILE_PATH_SEPARATOR_REGEX = /[\\/]/;

// Separates a file name from its extension, e.g. the '.' in `cover.png`.
export const FILE_EXTENSION_SEPARATOR = '.';

// Matches the `text:` prefix that file-type identifiers use to denote a
// plain-text language (e.g. `text:typescript`). Used by tool-call renderers
// to recover the underlying highlight.js language.
export const TEXT_LANGUAGE_PREFIX_REGEX = /^text:/;

/** CSS classes applied by the markdown code-block renderer. */
export const CODE_BLOCK_CLASS = {
	ACTIONS: 'code-block-actions',
	COPY_BTN: 'copy-code-btn',
	DOWNLOAD_BTN: 'download-code-btn',
	HEADER: 'code-block-header',
	LANGUAGE: 'code-language',
	PREVIEW_BTN: 'preview-code-btn',
	RELATIVE: 'relative',
	SCROLL_CONTAINER: 'code-block-scroll-container',
	WRAPPER: 'code-block-wrapper'
} as const;

/** Language sensitive texts */
export const CODE_BLOCK_TEXT = {
	COPY_BTN_TITLE: 'Copy code',
	DOWNLOAD_BTN_TITLE: 'Download file',
	PREVIEW_TITLE: 'Preview code'
} as const;

/** Markdown code block type name to file extension mapping */
export const CODE_BLOCK_TYPE_TO_EXTENSION_MAP: Record<string, string> = {
	// A
	actionscript: '.as',
	ada: '.adb',
	antlr4: '.g4',
	apache: '.conf',
	apacheconf: '.conf',
	apex: '.cls',
	arduino: '.ino',
	'arm-asm': '.s',
	armasm: '.s',
	arturo: '.art',
	asciidoc: '.adoc',
	asm6502: '.asm',
	asmatmel: '.asm',
	aspnet: '.aspx',
	autohotkey: '.ahk',
	autoit: '.au3',
	avisynth: '.avs',
	'avro-idl': '.avdl',

	// B
	bash: '.sh',
	batch: '.bat',
	bison: '.y',
	brainfuck: '.bf',
	brightscript: '.brs',

	// C
	'c#': '.cs',
	'c++': '.cpp',
	cbl: '.cob',
	cc: '.cpp',
	cfscript: '.cfc',
	chaiscript: '.chai',
	cil: '.il',
	clojure: '.clj',
	cmd: '.bat',
	cobol: '.cob',
	coffeescript: '.coffee',
	'common-lisp': '.lisp',
	concurnas: '.conc',
	coq: '.v',
	crystal: '.cr',
	csharp: '.cs',
	cxx: '.cpp',
	cypher: '.cql',

	// D
	dataweave: '.dwl',
	django: '.jinja',
	docker: '.dockerfile',
	dockerfile: '.dockerfile',
	dos: '.bat',
	dot: '.gv',

	// E
	eiffel: '.e',
	elisp: '.el',
	elixir: '.ex',
	'emacs-lisp': '.el',
	erlang: '.erl',
	'excel-formula': '.xlsx',

	// F
	'f#': '.fs',
	faust: '.dsp',
	flow: '.js',
	fortran: '.f90',
	freemarker: '.ftl',
	fsharp: '.fs',

	// G
	gawk: '.awk',
	gdscript: '.gd',
	gedcom: '.ged',
	gherkin: '.feature',
	git: '.gitignore',
	golang: '.go',
	gql: '.graphql',
	gvy: '.groovy',

	// H
	handlebars: '.hbs',
	haskell: '.hs',
	haxe: '.hx',
	hcl: '.tf',

	// I
	ichigojam: '.bas',
	icon: '.icn',
	idris: '.idr',
	iecst: '.st',
	inform7: '.ni',

	// J
	j: '.ijs',
	jade: '.pug',
	javadoc: '.java',
	javascript: '.js',
	jinja: '.j2',
	jinja2: '.j2',
	jolie: '.ol',
	jsonp: '.js',
	julia: '.jl',

	// K
	keepalived: '.conf',
	keyman: '.kmn',
	kotlin: '.kt',
	kusto: '.kql',

	// L
	latex: '.tex',
	lilypond: '.ly',
	livescript: '.ls',
	llvm: '.ll',
	lolcode: '.lol',

	// M
	make: '.mk',
	makefile: '.mk',
	markdown: '.md',
	markup: '.html',
	mathematica: '.nb',
	matlab: '.m',
	maxscript: '.ms',
	mermaid: '.mmd',
	mizar: '.miz',
	mongodb: '.js',
	moonscript: '.moon',
	mustache: '.hbs',

	// N
	nasm: '.asm',
	nginx: '.conf',
	node: '.js',
	nsis: '.nsi',

	// O
	'obj-c': '.m',
	objc: '.m',
	'objective-c': '.m',
	'objective-cpp': '.mm',
	objectivec: '.m',
	objectpascal: '.pas',
	ocaml: '.ml',
	opencl: '.cl',
	openqasm: '.qasm',
	oscript: '.bsl',

	// P
	parigp: '.gp',
	pascal: '.pas',
	pawn: '.pwn',
	perl: '.pl',
	plain: '.txt',
	plaintext: '.txt',
	plantuml: '.puml',
	plsql: '.pls',
	powerquery: '.pq',
	powershell: '.ps1',
	processing: '.pde',
	prolog: '.pl',
	protobuf: '.proto',
	puppet: '.pp',
	purebasic: '.pb',
	purescript: '.purs',
	python: '.py',

	// Q
	qore: '.q',
	qsharp: '.qs',

	// R
	racket: '.rkt',
	razor: '.cshtml',
	react: '.jsx',
	reason: '.re',
	renpy: '.rpy',
	rescript: '.res',
	rest: '.http',
	restructuredtext: '.rst',
	rmarkdown: '.Rmd',
	robotframework: '.robot',
	ruby: '.rb',
	rust: '.rs',

	// S
	scheme: '.scm',
	shell: '.sh',
	smalltalk: '.st',
	sparql: '.rq',
	'splunk-spl': '.spl',
	squirrel: '.nut',
	stata: '.do',
	stylus: '.styl',
	systemd: '.service',

	// T
	t4: '.tt',
	't4-cs': '.tt',
	't4-vb': '.tt',
	terraform: '.tf',
	text: '.txt',
	typescript: '.ts',

	// U
	unison: '.u',

	// V
	vba: '.bas',
	vbnet: '.vb',
	vbscript: '.vbs',
	velocity: '.vm',
	verilog: '.v',
	vhdl: '.vhd',
	viml: '.vim',
	'visual-basic': '.vb',
	visualbasic: '.vb',

	// W
	webassembly: '.wasm',
	wolfram: '.nb',

	// X, Y, Z
	xquery: '.xq',
	yaml: '.yml',
	zsh: '.sh'
};
