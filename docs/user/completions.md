# Completions

Command-line completion is available for some environments.

## Bash Completion

```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```

Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:

```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```
