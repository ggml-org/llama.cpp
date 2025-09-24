import { toast } from 'svelte-sonner';

/**
 * Copy text to clipboard with toast notification
 * Uses modern clipboard API when available, falls back to legacy method for non-secure contexts
 * @param text - Text to copy to clipboard
 * @param successMessage - Custom success message (optional)
 * @param errorMessage - Custom error message (optional)
 * @returns Promise<boolean> - True if successful, false otherwise
 */
export async function copyToClipboard(
	text: string,
	successMessage = 'Copied to clipboard',
	errorMessage = 'Failed to copy to clipboard'
): Promise<boolean> {
	try {
		// Try modern clipboard API first (secure contexts only)
		if (navigator.clipboard && navigator.clipboard.writeText) {
			await navigator.clipboard.writeText(text);
			toast.success(successMessage);
			return true;
		}

		// Fallback for non-secure contexts
		const textArea = document.createElement('textarea');
		textArea.value = text;
		textArea.style.position = 'fixed';
		textArea.style.left = '-999999px';
		textArea.style.top = '-999999px';
		document.body.appendChild(textArea);
		textArea.focus();
		textArea.select();

		const successful = document.execCommand('copy');
		document.body.removeChild(textArea);

		if (successful) {
			toast.success(successMessage);
			return true;
		} else {
			throw new Error('execCommand failed');
		}
	} catch (error) {
		console.error('Failed to copy to clipboard:', error);
		toast.error(errorMessage);
		return false;
	}
}

/**
 * Copy code with HTML entity decoding and toast notification
 * @param rawCode - Raw code string that may contain HTML entities
 * @param successMessage - Custom success message (optional)
 * @param errorMessage - Custom error message (optional)
 * @returns Promise<boolean> - True if successful, false otherwise
 */
export async function copyCodeToClipboard(
	rawCode: string,
	successMessage = 'Code copied to clipboard',
	errorMessage = 'Failed to copy code'
): Promise<boolean> {
	// Decode HTML entities
	const decodedCode = rawCode
		.replace(/&amp;/g, '&')
		.replace(/&lt;/g, '<')
		.replace(/&gt;/g, '>')
		.replace(/&quot;/g, '"')
		.replace(/&#39;/g, "'");

	return copyToClipboard(decodedCode, successMessage, errorMessage);
}
