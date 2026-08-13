/**
 * Comprehensive dictionary of all supported file types in llama-ui
 * Organized by category with TypeScript enums for better type safety
 */

import {
	FileExtensionAudio,
	FileExtensionImage,
	FileExtensionPdf,
	FileExtensionText,
	FileTypeAudio,
	FileTypeImage,
	FileTypePdf,
	FileTypeText,
	MimeTypeApplication,
	MimeTypeAudio,
	MimeTypeImage,
	MimeTypeText,
	MimeTypeVideo
} from '$lib/enums';
import { FileExtensionVideo, FileTypeVideo } from '$lib/enums/files.enums';

// File type configuration using enums
export const AUDIO_FILE_TYPES = {
	[FileTypeAudio.MP3]: {
		extensions: [FileExtensionAudio.MP3],
		mimeTypes: [MimeTypeAudio.MP3_MPEG, MimeTypeAudio.MP3]
	},
	[FileTypeAudio.WAV]: {
		extensions: [FileExtensionAudio.WAV],
		mimeTypes: [MimeTypeAudio.WAV]
	}
} as const;

export const VIDEO_FILE_TYPES = {
	[FileTypeVideo.MP4]: {
		extensions: [FileExtensionVideo.MP4],
		mimeTypes: [MimeTypeVideo.MP4]
	},
	[FileTypeVideo.OGG]: {
		extensions: [FileExtensionVideo.OGG],
		mimeTypes: [MimeTypeVideo.OGG]
	}
} as const;

export const IMAGE_FILE_TYPES = {
	[FileTypeImage.GIF]: {
		extensions: [FileExtensionImage.GIF],
		mimeTypes: [MimeTypeImage.GIF]
	},
	[FileTypeImage.HEIC]: {
		extensions: [FileExtensionImage.HEIC, FileExtensionImage.HEIF],
		mimeTypes: [MimeTypeImage.HEIC, MimeTypeImage.HEIF]
	},
	[FileTypeImage.JPEG]: {
		extensions: [FileExtensionImage.JPG, FileExtensionImage.JPEG],
		mimeTypes: [MimeTypeImage.JPEG]
	},
	[FileTypeImage.PNG]: {
		extensions: [FileExtensionImage.PNG],
		mimeTypes: [MimeTypeImage.PNG]
	},
	[FileTypeImage.SVG]: {
		extensions: [FileExtensionImage.SVG],
		mimeTypes: [MimeTypeImage.SVG]
	},
	[FileTypeImage.WEBP]: {
		extensions: [FileExtensionImage.WEBP],
		mimeTypes: [MimeTypeImage.WEBP]
	}
} as const;

export const PDF_FILE_TYPES = {
	[FileTypePdf.PDF]: {
		extensions: [FileExtensionPdf.PDF],
		mimeTypes: [MimeTypeApplication.PDF]
	}
} as const;

export const TEXT_FILE_TYPES = {
	[FileTypeText.ASCIIDOC]: {
		extensions: [FileExtensionText.ADOC],
		mimeTypes: [MimeTypeText.ASCIIDOC]
	},
	[FileTypeText.BIBTEX]: {
		extensions: [FileExtensionText.BIB],
		mimeTypes: [MimeTypeText.BIBTEX]
	},
	[FileTypeText.CPP]: {
		extensions: [
			FileExtensionText.CPP,
			FileExtensionText.C,
			FileExtensionText.H,
			FileExtensionText.HPP
		],
		mimeTypes: [MimeTypeText.CPP_SRC, MimeTypeText.CPP_HDR, MimeTypeText.C_SRC, MimeTypeText.C_HDR]
	},
	[FileTypeText.CSHARP]: {
		extensions: [FileExtensionText.CS],
		mimeTypes: [MimeTypeText.CSHARP]
	},
	[FileTypeText.CSS]: {
		extensions: [FileExtensionText.CSS],
		mimeTypes: [MimeTypeText.CSS]
	},
	[FileTypeText.CSV]: {
		extensions: [FileExtensionText.CSV],
		mimeTypes: [MimeTypeText.CSV]
	},
	[FileTypeText.CUDA]: {
		extensions: [FileExtensionText.CU, FileExtensionText.CUH],
		mimeTypes: [MimeTypeText.CUDA]
	},
	[FileTypeText.DART]: {
		extensions: [FileExtensionText.DART],
		mimeTypes: [MimeTypeText.DART]
	},
	[FileTypeText.GO]: {
		extensions: [FileExtensionText.GO],
		mimeTypes: [MimeTypeText.GO]
	},
	[FileTypeText.HASKELL]: {
		extensions: [FileExtensionText.HS],
		mimeTypes: [MimeTypeText.HASKELL]
	},
	[FileTypeText.HTML]: {
		extensions: [FileExtensionText.HTML, FileExtensionText.HTM],
		mimeTypes: [MimeTypeText.HTML]
	},
	[FileTypeText.JAVA]: {
		extensions: [FileExtensionText.JAVA],
		mimeTypes: [MimeTypeText.JAVA]
	},
	[FileTypeText.JAVASCRIPT]: {
		extensions: [FileExtensionText.JS],
		mimeTypes: [MimeTypeText.JAVASCRIPT, MimeTypeText.JAVASCRIPT_APP]
	},
	[FileTypeText.JSON]: {
		extensions: [FileExtensionText.JSON],
		mimeTypes: [MimeTypeText.JSON]
	},
	[FileTypeText.JSX]: {
		extensions: [FileExtensionText.JSX],
		mimeTypes: [MimeTypeText.JSX]
	},
	[FileTypeText.KOTLIN]: {
		extensions: [FileExtensionText.KT],
		mimeTypes: [MimeTypeText.KOTLIN]
	},
	[FileTypeText.LATEX]: {
		extensions: [FileExtensionText.TEX],
		mimeTypes: [MimeTypeText.LATEX, MimeTypeText.TEX, MimeTypeText.TEX_APP]
	},
	[FileTypeText.LOG]: {
		extensions: [FileExtensionText.LOG],
		mimeTypes: [MimeTypeText.PLAIN]
	},
	[FileTypeText.MARKDOWN]: {
		extensions: [FileExtensionText.MD],
		mimeTypes: [MimeTypeText.MARKDOWN]
	},
	[FileTypeText.PHP]: {
		extensions: [FileExtensionText.PHP],
		mimeTypes: [MimeTypeText.PHP]
	},
	[FileTypeText.PLAIN_TEXT]: {
		extensions: [FileExtensionText.TXT],
		mimeTypes: [MimeTypeText.PLAIN]
	},
	[FileTypeText.PROPERTIES]: {
		extensions: [FileExtensionText.PROPERTIES],
		mimeTypes: [MimeTypeText.PROPERTIES]
	},
	[FileTypeText.PYTHON]: {
		extensions: [FileExtensionText.PY],
		mimeTypes: [MimeTypeText.PYTHON]
	},
	[FileTypeText.R]: {
		extensions: [FileExtensionText.R],
		mimeTypes: [MimeTypeText.R]
	},
	[FileTypeText.RUBY]: {
		extensions: [FileExtensionText.RB],
		mimeTypes: [MimeTypeText.RUBY]
	},
	[FileTypeText.RUST]: {
		extensions: [FileExtensionText.RS],
		mimeTypes: [MimeTypeText.RUST]
	},
	[FileTypeText.SCALA]: {
		extensions: [FileExtensionText.SCALA],
		mimeTypes: [MimeTypeText.SCALA]
	},
	[FileTypeText.SHELL]: {
		extensions: [FileExtensionText.SH, FileExtensionText.BAT],
		mimeTypes: [MimeTypeText.SHELL, MimeTypeText.BAT]
	},
	[FileTypeText.SQL]: {
		extensions: [FileExtensionText.SQL],
		mimeTypes: [MimeTypeText.SQL]
	},
	[FileTypeText.SVELTE]: {
		extensions: [FileExtensionText.SVELTE],
		mimeTypes: [MimeTypeText.SVELTE]
	},
	[FileTypeText.SWIFT]: {
		extensions: [FileExtensionText.SWIFT],
		mimeTypes: [MimeTypeText.SWIFT]
	},
	[FileTypeText.TSX]: {
		extensions: [FileExtensionText.TSX],
		mimeTypes: [MimeTypeText.TSX]
	},
	[FileTypeText.TYPESCRIPT]: {
		extensions: [FileExtensionText.TS],
		mimeTypes: [MimeTypeText.TYPESCRIPT]
	},
	[FileTypeText.VUE]: {
		extensions: [FileExtensionText.VUE],
		mimeTypes: [MimeTypeText.VUE]
	},
	[FileTypeText.VULKAN]: {
		extensions: [FileExtensionText.COMP],
		mimeTypes: [MimeTypeText.PLAIN]
	},
	[FileTypeText.XML]: {
		extensions: [FileExtensionText.XML],
		mimeTypes: [MimeTypeText.XML_TEXT, MimeTypeText.XML_APP]
	},
	[FileTypeText.YAML]: {
		extensions: [FileExtensionText.YAML, FileExtensionText.YML],
		mimeTypes: [MimeTypeText.YAML_TEXT, MimeTypeText.YAML_APP]
	}
} as const;
