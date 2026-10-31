Magic File for Model Information
------------------------

Until these models are registered with MIME/Media info you can add this to your local magic file to get simple model details without a hex editor.

### Prepare:
Magic local data for file(1) command. Append this to your /etc/magic file. Insert here your local magic data. Format is described in magic(5). Hopefully this will reach upstream/mainline magic files for distributions soon.

```
0	string	tjgg	GGML/GGJT LLM model
>0x4	lelong	<255	version=%d
0	string	GGUF	GGUF LLM model
>0x4	lelong	<255	version=%d
```
