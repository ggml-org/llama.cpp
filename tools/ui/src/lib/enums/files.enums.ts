/**
 * Comprehensive dictionary of all supported file types in llama-ui
 * Organized by category with TypeScript enums for better type safety
 */

// File type category enum
export enum FileTypeCategory {
	AUDIO = 'audio',
	IMAGE = 'image',
	PDF = 'pdf',
	TEXT = 'text',
	VIDEO = 'video'
}

/**
 * Special file types for internal use (not MIME types)
 */
export enum SpecialFileType {
	MCP_PROMPT = 'mcp-prompt'
}

// Specific file type enums for each category
export enum FileTypeImage {
	GIF = 'gif',
	HEIC = 'heic',
	HEIF = 'heif',
	JPEG = 'jpeg',
	PNG = 'png',
	SVG = 'svg',
	WEBP = 'webp'
}

export enum FileTypeAudio {
	MP3 = 'mp3',
	WAV = 'wav',
	WEBM = 'webm'
}

export enum FileTypeVideo {
	MP4 = 'mp4',
	OGG = 'ogg'
}

export enum FileTypePdf {
	PDF = 'pdf'
}

export enum FileTypeText {
	ASCIIDOC = 'asciidoc',
	BIBTEX = 'bibtex',
	CPP = 'cpp',
	CSHARP = 'csharp',
	CSS = 'css',
	CSV = 'csv',
	CUDA = 'cuda',
	DART = 'dart',
	GO = 'go',
	HASKELL = 'haskell',
	HTML = 'html',
	JAVA = 'java',
	JAVASCRIPT = 'js',
	JSON = 'json',
	JSX = 'jsx',
	KOTLIN = 'kotlin',
	LATEX = 'latex',
	LOG = 'log',
	MARKDOWN = 'md',
	PHP = 'php',
	PLAIN_TEXT = 'plainText',
	PROPERTIES = 'properties',
	PYTHON = 'python',
	R = 'r',
	RUBY = 'ruby',
	RUST = 'rust',
	SCALA = 'scala',
	SHELL = 'shell',
	SQL = 'sql',
	SVELTE = 'svelte',
	SWIFT = 'swift',
	TSX = 'tsx',
	TYPESCRIPT = 'ts',
	VUE = 'vue',
	VULKAN = 'vulkan',
	XML = 'xml',
	YAML = 'yaml'
}

// File extension enums
export enum FileExtensionImage {
	GIF = '.gif',
	HEIC = '.heic',
	HEIF = '.heif',
	JPEG = '.jpeg',
	JPG = '.jpg',
	PNG = '.png',
	SVG = '.svg',
	WEBP = '.webp'
}

export enum FileExtensionAudio {
	MP3 = '.mp3',
	WAV = '.wav'
}

export enum FileExtensionVideo {
	MP4 = '.mp4',
	OGG = '.ogg'
}

export enum FileExtensionPdf {
	PDF = '.pdf'
}

export enum FileExtensionText {
	ADOC = '.adoc',
	BAT = '.bat',
	BIB = '.bib',
	C = '.c',
	COMP = '.comp',
	CPP = '.cpp',
	CS = '.cs',
	CSS = '.css',
	CSV = '.csv',
	CU = '.cu',
	CUH = '.cuh',
	DART = '.dart',
	GO = '.go',
	H = '.h',
	HPP = '.hpp',
	HS = '.hs',
	HTM = '.htm',
	HTML = '.html',
	JAVA = '.java',
	JS = '.js',
	JSON = '.json',
	JSONL = '.jsonl',
	JSX = '.jsx',
	KT = '.kt',
	LOG = '.log',
	MD = '.md',
	PHP = '.php',
	PROPERTIES = '.properties',
	PY = '.py',
	R = '.r',
	RB = '.rb',
	RS = '.rs',
	SCALA = '.scala',
	SH = '.sh',
	SQL = '.sql',
	SVELTE = '.svelte',
	SWIFT = '.swift',
	TEX = '.tex',
	TS = '.ts',
	TSX = '.tsx',
	TXT = '.txt',
	VUE = '.vue',
	XML = '.xml',
	YAML = '.yaml',
	YML = '.yml',
	ZIP = '.zip'
}

// MIME type prefixes and includes for content detection
export enum MimeTypePrefix {
	AUDIO = 'audio/',
	IMAGE = 'image/',
	TEXT = 'text'
}

export enum MimeTypeIncludes {
	JAVASCRIPT = 'javascript',
	JSON = 'json',
	TYPESCRIPT = 'typescript'
}

// URI patterns for content detection
export enum UriPattern {
	DATABASE_KEYWORD = 'database',
	DATABASE_SCHEME = 'db://'
}

// MIME type enums
export enum MimeTypeApplication {
	JSON = 'application/json',
	OCTET_STREAM = 'application/octet-stream',
	PDF = 'application/pdf',
	ZIP = 'application/zip'
}

export enum MimeTypeAudio {
	MP3 = 'audio/mp3',
	MP3_MPEG = 'audio/mpeg',
	MP4 = 'audio/mp4',
	VND_WAVE = 'audio/vnd.wave',
	WAV = 'audio/wav',
	WAVE = 'audio/wave',
	WEBM = 'audio/webm',
	WEBM_OPUS = 'audio/webm;codecs=opus',
	X_PN_WAV = 'audio/x-pn-wav',
	X_WAV = 'audio/x-wav',
	X_WAVE = 'audio/x-wave'
}

export enum MimeTypeVideo {
	MP4 = 'video/mp4',
	OGG = 'video/ogg'
}

export enum MimeTypeImage {
	GIF = 'image/gif',
	HEIC = 'image/heic',
	HEIF = 'image/heif',
	ICO = 'image/x-icon',
	ICO_MICROSOFT = 'image/vnd.microsoft.icon',
	JPEG = 'image/jpeg',
	JPG = 'image/jpg',
	PNG = 'image/png',
	SVG = 'image/svg+xml',
	WEBP = 'image/webp'
}

export enum MimeTypeText {
	ASCIIDOC = 'text/asciidoc',
	BAT = 'application/x-bat',
	BIBTEX = 'text/x-bibtex',
	C_HDR = 'text/x-chdr',
	C_SRC = 'text/x-csrc',
	CPP_HDR = 'text/x-c++hdr',
	CPP_SRC = 'text/x-c++src',
	CSHARP = 'text/x-csharp',
	CSS = 'text/css',
	CSV = 'text/csv',
	CUDA = 'text/x-cuda',
	DART = 'text/x-dart',
	GO = 'text/x-go',
	HASKELL = 'text/x-haskell',
	HTML = 'text/html',
	JAVA = 'text/x-java-source',
	JAVASCRIPT = 'text/javascript',
	JAVASCRIPT_APP = 'application/javascript',
	JSON = 'application/json',
	JSONL = 'application/jsonl',
	JSX = 'text/jsx',
	KOTLIN = 'text/x-kotlin',
	LATEX = 'application/x-latex',
	MARKDOWN = 'text/markdown',
	PHP = 'text/x-php',
	PLAIN = 'text/plain',
	PROPERTIES = 'text/properties',
	PYTHON = 'text/x-python',
	R = 'text/x-r',
	RUBY = 'text/x-ruby',
	RUST = 'text/x-rust',
	SCALA = 'text/x-scala',
	SHELL = 'text/x-shellscript',
	SQL = 'text/x-sql',
	SVELTE = 'text/x-svelte',
	SWIFT = 'text/x-swift',
	TEX = 'text/x-tex',
	TEX_APP = 'application/x-tex',
	TSX = 'text/tsx',
	TYPESCRIPT = 'text/typescript',
	VUE = 'text/x-vue',
	XML_APP = 'application/xml',
	XML_TEXT = 'text/xml',
	YAML_APP = 'application/yaml',
	YAML_TEXT = 'text/yaml'
}
