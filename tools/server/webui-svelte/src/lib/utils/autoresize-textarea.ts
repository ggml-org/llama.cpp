export default function autoResizeTextarea(textareaElement: HTMLTextAreaElement) {
	if (textareaElement) {
		textareaElement.style.height = 'auto';
		textareaElement.style.height = textareaElement.scrollHeight + 'px';
	}
}
