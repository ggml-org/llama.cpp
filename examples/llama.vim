" LLM-based text completion using llama.cpp
"
" requires:
"
"   - neovim or vim
"   - curl
"   - llama.cpp server instance
"   - FIM-compatible model
"
" sample config:
"
"   - Tab       - accept the current suggestion
"   - Shift+Tab - accept just the first line of the suggestion
"   - Ctrl+F    - toggle FIM completion manually
"
" make symlink or copy this file to ~/.config/nvim/autoload/llama.vim
"
" start the llama.cpp server with a FIM-compatible model. for example:
"
"   $ llama-server -m {model.gguf} --port 8012 -ngl 99 -fa -dt 0.1 --ubatch-size 512 --batch-size 1024 --cache-reuse 256
"
"   --batch-size [512, model max context]
"
"     adjust the batch size to control how much of the provided local context will be used during the inference
"     lower values will use smaller part of the context around the cursor, which will result in faster processing
"
"   --ubatch-size [64, 2048]
"
"     chunks the batch into smaller chunks for faster processing
"     depends on the specific hardware. use llama-bench to profile and determine the best size
"
"   --cache-reuse (ge:llama_config.n_predict, 1024]
"
"     this should be either 0 (disabled) or strictly larger than g:llama_config.n_predict
"     using non-zero value enables context reuse on the server side which dramatically improves the performance at
"     large contexts. a value of 256 should be good for all cases
"
" run this once to initialise llama.vim:
"
"   :call llama#init()
"
" more info: https://github.com/ggml-org/llama.cpp/pull/9787
"

" colors (adjust to your liking)
highlight llama_hl_hint guifg=#ff772f ctermfg=202
highlight llama_hl_info guifg=#77ff2f ctermfg=119

" general parameters:
"
"   endpoint:         llama.cpp server endpoint
"   n_prefix:         number of lines before the cursor location to include in the local prefix
"   n_suffix:         number of lines after  the cursor location to include in the local suffix
"   n_predict:        max number of tokens to predict
"   t_max_prompt_ms:  max alloted time for the prompt processing (TODO: not yet supported)
"   t_max_predict_ms: max alloted time for the prediction
"   show_info:        show extra info about the inference (0 - disabled, 1 - statusline, 2 - inline)
"   auto_fim:         trigger FIM completion automatically on cursor movement
"   max_line_suffix:  do not auto-trigger FIM completion if there are more than this number of characters to the right of the cursor
"
" ring buffer of chunks, accumulated with time upon:
"
"  - completion request
"  - yank
"  - entering a buffer
"  - leaving a buffer
"  - writing a file
"
" parameters for the ring-buffer with extra context:
"
"   ring_n_chunks:    max number of chunks to pass as extra context to the server (0 to disable)
"   ring_chunk_size:  max size of the chunks (in number of lines)
"                     note: adjust these numbers so that you don't overrun your context
"                           at ring_n_chunks = 64 and ring_chunk_size = 64 you need ~32k context
"   ring_scope:       the range around the cursor position (in number of lines) for gathering chunks after FIM
"   ring_update_ms:   how often to process queued chunks in normal mode
"
let s:default_config = {
    \ 'endpoint':         'http://127.0.0.1:8012/infill',
    \ 'n_prefix':         256,
    \ 'n_suffix':         64,
    \ 'n_predict':        128,
    \ 't_max_prompt_ms':  500,
    \ 't_max_predict_ms': 3000,
    \ 'show_info':        2,
    \ 'auto_fim':         v:true,
    \ 'max_line_suffix':  8,
    \ 'ring_n_chunks':    64,
    \ 'ring_chunk_size':  64,
    \ 'ring_scope':       1024,
    \ 'ring_update_ms':   1000,
    \ }

let g:llama_config = get(g:, 'llama_config', s:default_config)

function! s:get_indent(str)
    let l:count = 0
    for i in range(len(a:str))
        if a:str[i] == "\t"
            let l:count += &tabstop - 1
        else
            break
        endif
    endfor
    return l:count
endfunction

function! s:rand(i0, i1) abort
    return a:i0 + rand() % (a:i1 - a:i0 + 1)
endfunction

function! llama#init()
    if !executable('curl')
        echohl WarningMsg
        echo 'llama.vim requires the "curl" command to be available'
        echohl None
        return
    endif

    let s:pos_x = 0 " cursor position upon start of completion
    let s:pos_y = 0

    let s:line_cur = ''

    let s:line_cur_prefix = ''
    let s:line_cur_suffix = ''

    let s:ring_chunks = [] " current set of chunks used as extra context
    let s:ring_queued = [] " chunks that are queued to be sent for processing
    let s:ring_n_evict = 0

    let s:hint_shown = v:false
    let s:pos_y_pick = -9999 " last y where we picked a chunk
    let s:pos_dx = 0
    let s:content = []
    let s:can_accept = v:false

    let s:timer_fim = -1
    let s:t_fim_start = reltime() " used to measure total FIM time
    let s:t_last_move = reltime() " last time the cursor moved

    let s:current_job = v:null

    let s:ghost_text_nvim = exists('*nvim_buf_get_mark')
    let s:ghost_text_vim = has('textprop')

    if s:ghost_text_vim
        let s:hlgroup_hint = 'llama_hl_hint'
        let s:hlgroup_info = 'llama_hl_info'

        if empty(prop_type_get(s:hlgroup_hint))
            call prop_type_add(s:hlgroup_hint, {'highlight': s:hlgroup_hint})
        endif
        if empty(prop_type_get(s:hlgroup_info))
            call prop_type_add(s:hlgroup_info, {'highlight': s:hlgroup_info})
        endif
    endif

    augroup llama
        autocmd!
        autocmd InsertEnter     * inoremap <expr> <silent> <C-F> llama#fim_inline(v:false)
        autocmd InsertLeavePre  * call llama#fim_cancel()

        autocmd CursorMoved     * call s:on_move()
        autocmd CursorMovedI    * call s:on_move()
        autocmd CompleteChanged * call llama#fim_cancel()

        if g:llama_config.auto_fim
            autocmd CursorMovedI * call llama#fim(v:true)
        endif

        " gather chunks upon yanking
        autocmd TextYankPost    * if v:event.operator ==# 'y' | call s:pick_chunk(v:event.regcontents, v:false, v:true) | endif

        " gather chunks upon entering/leaving a buffer
        autocmd BufEnter        * call timer_start(100, {-> s:pick_chunk(getline(max([1, line('.') - g:llama_config.ring_chunk_size/2]), min([line('.') + g:llama_config.ring_chunk_size/2, line('$')])), v:true, v:true)})
        autocmd BufLeave        * call                      s:pick_chunk(getline(max([1, line('.') - g:llama_config.ring_chunk_size/2]), min([line('.') + g:llama_config.ring_chunk_size/2, line('$')])), v:true, v:true)

        " gather chunk upon saving the file
        autocmd BufWritePost    * call s:pick_chunk(getline(max([1, line('.') - g:llama_config.ring_chunk_size/2]), min([line('.') + g:llama_config.ring_chunk_size/2, line('$')])), v:true, v:true)
    augroup END

    silent! call llama#fim_cancel()

    " init background update of the ring buffer
    if g:llama_config.ring_n_chunks > 0
        call s:ring_update()
    endif
endfunction

" compute how similar two chunks of text are
" 0 - no similarity, 1 - high similarity
" TODO: figure out something better
function! s:chunk_sim(c0, c1)
    let l:lines0 = len(a:c0)
    let l:lines1 = len(a:c1)

    let l:common = 0

    for l:line0 in a:c0
        for l:line1 in a:c1
            if l:line0 == l:line1
                let l:common += 1
                break
            endif
        endfor
    endfor

    return 2.0 * l:common / (l:lines0 + l:lines1)
endfunction

" pick a random chunk of size g:llama_config.ring_chunk_size from the provided text and queue it for processing
"
" no_mod   - do not pick chunks from buffers with pending changes
" do_evict - evict chunks that are very similar to the new one
"
function! s:pick_chunk(text, no_mod, do_evict)
    " do not pick chunks from buffers with pending changes or buffers that are not files
    if a:no_mod && (getbufvar(bufnr('%'), '&modified') || !buflisted(bufnr('%')) || !filereadable(expand('%')))
        return
    endif

    " if the extra context option is disabled - do nothing
    if g:llama_config.ring_n_chunks <= 0
        return
    endif

    " don't pick very small chunks
    if len(a:text) < 3
        return
    endif

    if len(a:text) + 1 < g:llama_config.ring_chunk_size
        let l:chunk = a:text
    else
        let l:l0 = s:rand(0, max([0, len(a:text) - g:llama_config.ring_chunk_size/2]))
        let l:l1 = min([l:l0 + g:llama_config.ring_chunk_size/2, len(a:text)])

        let l:chunk = a:text[l:l0:l:l1]
    endif

    let l:chunk_str = join(l:chunk, "\n") . "\n"

    " check if this chunk is already added
    let l:exist = v:false

    for i in range(len(s:ring_chunks))
        if s:ring_chunks[i].data == l:chunk
            let l:exist = v:true
            break
        endif
    endfor

    for i in range(len(s:ring_queued))
        if s:ring_queued[i].data == l:chunk
            let l:exist = v:true
            break
        endif
    endfor

    if l:exist
        return
    endif

    " evict queued chunks that are very similar to the new one
    for i in range(len(s:ring_queued) - 1, 0, -1)
        if s:chunk_sim(s:ring_queued[i].data, l:chunk) > 0.9
            if a:do_evict
                call remove(s:ring_queued, i)
                let s:ring_n_evict += 1
            else
                return
            endif
        endif
    endfor

    " also from s:ring_chunks
    for i in range(len(s:ring_chunks) - 1, 0, -1)
        if s:chunk_sim(s:ring_chunks[i].data, l:chunk) > 0.9
            if a:do_evict
                call remove(s:ring_chunks, i)
                let s:ring_n_evict += 1
            else
                return
            endif
        endif
    endfor

    " TODO: become parameter ?
    if len(s:ring_queued) == 16
        call remove(s:ring_queued, 0)
    endif

    call add(s:ring_queued, {'data': l:chunk, 'str': l:chunk_str, 'time': reltime(), 'filename': expand('%')})

    "let &statusline = 'extra context: ' . len(s:ring_chunks) . ' / ' . len(s:ring_queued)
endfunction

" picks a queued chunk, sends it for processing and adds it to s:ring_chunks
" called every g:llama_config.ring_update_ms
function! s:ring_update()
    call timer_start(g:llama_config.ring_update_ms, {-> s:ring_update()})

    " update only if in normal mode or if the cursor hasn't moved for a while
    if mode() !=# 'n' && reltimefloat(reltime(s:t_last_move)) < 3.0
        return
    endif

    if len(s:ring_queued) == 0
        return
    endif

    " move the first queued chunk to the ring buffer
    if len(s:ring_chunks) == g:llama_config.ring_n_chunks
        call remove(s:ring_chunks, 0)
    endif

    call add(s:ring_chunks, remove(s:ring_queued, 0))

    "let &statusline = 'updated context: ' . len(s:ring_chunks) . ' / ' . len(s:ring_queued)

    " send asynchronous job with the new extra context so that it is ready for the next FIM
    let l:extra_context = []
    for l:chunk in s:ring_chunks
        call add(l:extra_context, {
            \ 'text':     l:chunk.str,
            \ 'time':     l:chunk.time,
            \ 'filename': l:chunk.filename
            \ })
    endfor

    " no samplers needed here
    let l:request = json_encode({
        \ 'input_prefix':     "",
        \ 'input_suffix':     "",
        \ 'input_extra':      l:extra_context,
        \ 'prompt':           "",
        \ 'n_predict':        1,
        \ 'temperature':      0.0,
        \ 'stream':           v:false,
        \ 'samplers':         ["temperature"],
        \ 'cache_prompt':     v:true,
        \ 't_max_prompt_ms':  1,
        \ 't_max_predict_ms': 1
        \ })

    let l:curl_command = [
        \ "curl",
        \ "--silent",
        \ "--no-buffer",
        \ "--request", "POST",
        \ "--url", g:llama_config.endpoint,
        \ "--header", "Content-Type: application/json",
        \ "--data", l:request
        \ ]

    " no callbacks because we don't need to process the response
    if s:ghost_text_nvim
        call jobstart(l:curl_command, {})
    elseif s:ghost_text_vim
        call job_start(l:curl_command, {})
    endif
endfunction

" necessary for 'inoremap <expr>'
function! llama#fim_inline(is_auto) abort
    call llama#fim(a:is_auto)
    return ''
endfunction

" the main FIM call
" takes local context around the cursor and sends it together with the extra context to the server for completion
function! llama#fim(is_auto) abort
    " we already have a suggestion for the current cursor position
    if s:hint_shown && !a:is_auto
        call llama#fim_cancel()
        return
    endif

    call llama#fim_cancel()

    " avoid sending repeated requests too fast
    if reltimefloat(reltime(s:t_fim_start)) < 0.6
        if s:timer_fim != -1
            call timer_stop(s:timer_fim)
            let s:timer_fim = -1
        endif

        let s:t_fim_start = reltime()
        let s:timer_fim = timer_start(600, {-> llama#fim(v:true)})
        return
    endif

    let s:t_fim_start = reltime()

    let s:content = []
    let s:can_accept = v:false

    let s:pos_x = col('.') - 1
    let s:pos_y = line('.')
    let l:max_y = line('$')

    let l:lines_prefix = getline(max([1, s:pos_y - g:llama_config.n_prefix]), s:pos_y - 1)
    let l:lines_suffix = getline(s:pos_y + 1, min([l:max_y, s:pos_y + g:llama_config.n_suffix]))

    let s:line_cur = getline('.')

    let s:line_cur_prefix = strpart(s:line_cur, 0, s:pos_x)
    let s:line_cur_suffix = strpart(s:line_cur, s:pos_x)

    if a:is_auto && len(s:line_cur_suffix) > g:llama_config.max_line_suffix
        return
    endif

    let l:prefix = ""
        \ . join(l:lines_prefix, "\n")
        \ . "\n"

    let l:prompt = ""
        \ . s:line_cur_prefix

    let l:suffix = ""
        \ . s:line_cur_suffix
        \ . "\n"
        \ . join(l:lines_suffix, "\n")
        \ . "\n"

    " prepare the extra context data
    let l:extra_context = []
    for l:chunk in s:ring_chunks
        call add(l:extra_context, {
            \ 'text':     l:chunk.str,
            \ 'time':     l:chunk.time,
            \ 'filename': l:chunk.filename
            \ })
    endfor

    " the indentation of the current line
    let l:indent = strlen(matchstr(s:line_cur_prefix, '^\s*'))

    let l:request = json_encode({
        \ 'input_prefix':     l:prefix,
        \ 'input_suffix':     l:suffix,
        \ 'input_extra':      l:extra_context,
        \ 'prompt':           l:prompt,
        \ 'n_predict':        g:llama_config.n_predict,
        \ 'n_indent':         l:indent,
        \ 'top_k':            40,
        \ 'top_p':            0.99,
        \ 'stream':           v:false,
        \ 'samplers':         ["top_k", "top_p", "infill"],
        \ 'cache_prompt':     v:true,
        \ 't_max_prompt_ms':  g:llama_config.t_max_prompt_ms,
        \ 't_max_predict_ms': g:llama_config.t_max_predict_ms
        \ })

    let l:curl_command = [
        \ "curl",
        \ "--silent",
        \ "--no-buffer",
        \ "--request", "POST",
        \ "--url", g:llama_config.endpoint,
        \ "--header", "Content-Type: application/json",
        \ "--data", l:request
        \ ]

    if s:current_job != v:null
        if s:ghost_text_nvim
            call jobstop(s:current_job)
        elseif s:ghost_text_vim
            call job_stop(s:current_job)
        endif
    endif

    " send the request asynchronously
    if s:ghost_text_nvim
        let s:current_job = jobstart(l:curl_command, {
            \ 'on_stdout': function('s:fim_on_stdout', [s:pos_x, s:pos_y, a:is_auto]),
            \ 'on_exit':   function('s:fim_on_exit'),
            \ 'stdout_buffered': v:true
            \ })
    elseif s:ghost_text_vim
        let s:current_job = job_start(l:curl_command, {
            \ 'out_cb': function('s:fim_on_stdout', [s:pos_x, s:pos_y, a:is_auto]),
            \ 'exit_cb':   function('s:fim_on_exit')
            \ })
    endif

    " TODO: per-file location
    let l:delta_y = abs(s:pos_y - s:pos_y_pick)

    " gather some extra context nearby and process it in the background
    " only gather chunks if the cursor has moved a lot
    " TODO: something more clever? reranking?
    if a:is_auto && l:delta_y > 32
        " expand the prefix even further
        call s:pick_chunk(getline(max([1,       s:pos_y - g:llama_config.ring_scope]), max([1,       s:pos_y - g:llama_config.n_prefix])), v:false, v:false)

        " pick a suffix chunk
        call s:pick_chunk(getline(min([l:max_y, s:pos_y + g:llama_config.n_suffix]),   min([l:max_y, s:pos_y + g:llama_config.n_suffix + g:llama_config.ring_chunk_size])), v:false, v:false)

        let s:pos_y_pick = s:pos_y
    endif
endfunction

" if first_line == v:true accept only the first line of the response
function! llama#fim_accept(first_line)
    " insert the suggestion at the cursor location
    if s:can_accept && len(s:content) > 0
        call setline(s:pos_y, s:line_cur[:(s:pos_x - 1)] . s:content[0])
        if len(s:content) > 1
            if !a:first_line
                call append(s:pos_y, s:content[1:-1])
            endif
        endif

        " move the cursor to the end of the accepted text
        if !a:first_line && len(s:content) > 1
            call cursor(s:pos_y + len(s:content) - 1, s:pos_x + s:pos_dx + 1)
        else
            call cursor(s:pos_y, s:pos_x + len(s:content[0]))
        endif
    endif

    call llama#fim_cancel()
endfunction

function! llama#fim_cancel()
    let s:hint_shown = v:false

    " clear the virtual text
    let l:bufnr = bufnr('%')

    if s:ghost_text_nvim
        let l:id_vt_fim = nvim_create_namespace('vt_fim')
        call nvim_buf_clear_namespace(l:bufnr, l:id_vt_fim,  0, -1)
    elseif s:ghost_text_vim
        call prop_remove({'type': s:hlgroup_hint, 'all': v:true})
        call prop_remove({'type': s:hlgroup_info, 'all': v:true})
    endif

    " remove the mappings
    silent! iunmap <buffer> <Tab>
    silent! iunmap <buffer> <S-Tab>
    silent! iunmap <buffer> <Esc>
endfunction

function! s:on_move()
    let s:t_last_move = reltime()

    call llama#fim_cancel()
endfunction

" callback that processes the FIM result from the server and displays the suggestion
function! s:fim_on_stdout(pos_x, pos_y, is_auto, job_id, data, event = v:null)
    if s:ghost_text_nvim
        let l:raw = join(a:data, "\n")
    elseif s:ghost_text_vim
        let l:raw = a:data
    endif

    if len(l:raw) == 0
        return
    endif

    if a:pos_x != col('.') - 1 || a:pos_y != line('.')
        return
    endif

    " show the suggestion only in insert mode
    if mode() !=# 'i'
        return
    endif

    let s:pos_x = a:pos_x
    let s:pos_y = a:pos_y

    let s:can_accept = v:true
    let l:has_info   = v:false

    if s:can_accept && v:shell_error
        if !a:is_auto
            call add(s:content, "<| curl error: is the server on? |>")
        endif
        let s:can_accept = v:false
    endif

    let l:n_prompt    = 0
    let l:t_prompt_ms = 1.0
    let l:s_prompt    = 0

    let l:n_predict    = 0
    let l:t_predict_ms = 1.0
    let l:s_predict    = 0

    " get the generated suggestion
    if s:can_accept
        let l:response = json_decode(l:raw)

        for l:part in split(get(l:response, 'content', ''), "\n", 1)
            call add(s:content, l:part)
        endfor

        " remove trailing new lines
        while len(s:content) > 0 && s:content[-1] == ""
            call remove(s:content, -1)
        endwhile

        let l:generation_settings = get(l:response, 'generation_settings', {})
        let l:n_ctx = get(l:generation_settings, 'n_ctx', 0)

        let l:n_cached  = get(l:response, 'tokens_cached', 0)
        let l:truncated = get(l:response, 'truncated', v:false)

        " if response.timings is available
        if len(get(l:response, 'timings', {})) > 0
            let l:has_info = v:true
            let l:timings  = get(l:response, 'timings', {})

            let l:n_prompt    = get(l:timings, 'prompt_n', 0)
            let l:t_prompt_ms = get(l:timings, 'prompt_ms', 1)
            let l:s_prompt    = get(l:timings, 'prompt_per_second', 0)

            let l:n_predict    = get(l:timings, 'predicted_n', 0)
            let l:t_predict_ms = get(l:timings, 'predicted_ms', 1)
            let l:s_predict    = get(l:timings, 'predicted_per_second', 0)
        endif
    endif

    if len(s:content) == 0
        call add(s:content, "")
        let s:can_accept = v:false
    endif

    if len(s:content) == 0
        return
    endif

    " NOTE: the following is logic for discarding predictions that repeat existing text
    "       the code is quite ugly and there is very likely a simpler and more canonical way to implement this
    "
    "       still, I wonder if there is some better way that avoids having to do these special hacks?
    "       on one hand, the LLM 'sees' the contents of the file before we start editing, so it is normal that it would
    "       start generating whatever we have given it via the extra context. but on the other hand, it's not very
    "       helpful to re-generate the same code that is already there

    " truncate the suggestion if the first line is empty
    if len(s:content) == 1 && s:content[0] == ""
        let s:content = [""]
    endif

    " ... and the next lines are repeated
    if len(s:content) > 1 && s:content[0] == "" && s:content[1:] == getline(s:pos_y + 1, s:pos_y + len(s:content) - 1)
        let s:content = [""]
    endif

    " truncate the suggestion if it repeats the suffix
    if len(s:content) == 1 && s:content[0] == s:line_cur_suffix
        let s:content = [""]
    endif

    " find the first non-empty line (strip whitespace)
    let l:cmp_y = s:pos_y + 1
    while l:cmp_y < line('$') && getline(l:cmp_y) =~? '^\s*$'
        let l:cmp_y += 1
    endwhile

    if (s:line_cur_prefix . s:content[0]) == getline(l:cmp_y)
        " truncate the suggestion if it repeats the next line
        if len(s:content) == 1
            let s:content = [""]
        endif

        " ... or if the second line of the suggestion is the prefix of line l:cmp_y + 1
        if len(s:content) == 2 && s:content[-1] == getline(l:cmp_y + 1)[:len(s:content[-1]) - 1]
            let s:content = [""]
        endif

        " ... or if the middle chunk of lines of the suggestion is the same as [l:cmp_y + 1, l:cmp_y + len(s:content) - 1)
        if len(s:content) > 2 && join(s:content[1:-1], "\n") == join(getline(l:cmp_y + 1, l:cmp_y + len(s:content) - 1), "\n")
            let s:content = [""]
        endif
    endif

    " keep only lines that have the same or larger whitespace prefix as s:line_cur_prefix
    "let l:indent = strlen(matchstr(s:line_cur_prefix, '^\s*'))
    "for i in range(1, len(s:content) - 1)
    "    if strlen(matchstr(s:content[i], '^\s*')) < l:indent
    "        let s:content = s:content[:i - 1]
    "        break
    "    endif
    "endfor

    let s:pos_dx = len(s:content[-1])

    let s:content[-1] .= s:line_cur_suffix

    call llama#fim_cancel()

    " display virtual text with the suggestion
    let l:bufnr = bufnr('%')

    if s:ghost_text_nvim
        let l:id_vt_fim = nvim_create_namespace('vt_fim')
    endif

    " construct the info message
    if g:llama_config.show_info > 0 && l:has_info
        let l:prefix = '   '

        if l:truncated
            let l:info = printf("%s | WARNING: the context is full: %d / %d, increase the server context size or reduce g:llama_config.ring_n_chunks",
                \ g:llama_config.show_info == 2 ? l:prefix : 'llama.vim',
                \ l:n_cached, l:n_ctx
                \ )
        else
            let l:info = printf("%s | c: %d / %d, r: %d / %d, e: %d, q: %d / 16 | p: %d (%.2f ms, %.2f t/s) | g: %d (%.2f ms, %.2f t/s) | t: %.2f ms",
                \ g:llama_config.show_info == 2 ? l:prefix : 'llama.vim',
                \ l:n_cached,  l:n_ctx, len(s:ring_chunks), g:llama_config.ring_n_chunks, s:ring_n_evict, len(s:ring_queued),
                \ l:n_prompt,  l:t_prompt_ms,  l:s_prompt,
                \ l:n_predict, l:t_predict_ms, l:s_predict,
                \ 1000.0 * reltimefloat(reltime(s:t_fim_start))
                \ )
        endif

        if g:llama_config.show_info == 1
            " display the info in the statusline
            let &statusline = l:info
            let l:info = ''
        endif
    endif

    " display the suggestion and append the info to the end of the first line
    if s:ghost_text_nvim
        call nvim_buf_set_extmark(l:bufnr, l:id_vt_fim, s:pos_y - 1, s:pos_x - 1, {
            \ 'virt_text': [[s:content[0], 'llama_hl_hint'], [l:info, 'llama_hl_info']],
            \ 'virt_text_win_col': virtcol('.') - 1
            \ })

        call nvim_buf_set_extmark(l:bufnr, l:id_vt_fim, s:pos_y - 1, 0, {
            \ 'virt_lines': map(s:content[1:], {idx, val -> [[val, 'llama_hl_hint']]}),
            \ 'virt_text_win_col': virtcol('.')
            \ })
    elseif s:ghost_text_vim
        let l:new_suffix = s:content[0]
        if !empty(l:new_suffix)
            call prop_add(s:pos_y, s:pos_x + 1, {
                        \ 'type': s:hlgroup_hint,
                        \ 'text': l:new_suffix
                        \ })
        endif
        for line in s:content[1:]
            call prop_add(s:pos_y, 0, {
                        \ 'type': s:hlgroup_hint,
                        \ 'text': line,
                        \ 'text_padding_left': s:get_indent(line),
                        \ 'text_align': 'below'
                        \ })
        endfor
        if !empty(l:info)
            call prop_add(s:pos_y, 0, {
                        \ 'type': s:hlgroup_info,
                        \ 'text': l:info,
                        \ 'text_padding_left': col('$'),
                        \ 'text_wrap': 'truncate'
                        \ })
        endif
    endif

    " setup accept shortcuts
    inoremap <buffer> <Tab>   <C-O>:call llama#fim_accept(v:false)<CR>
    inoremap <buffer> <S-Tab> <C-O>:call llama#fim_accept(v:true)<CR>

    let s:hint_shown = v:true
endfunction

function! s:fim_on_exit(job_id, exit_code, event = v:null)
    if a:exit_code != 0
        echom "Job failed with exit code: " . a:exit_code
    endif

    let s:current_job = v:null
endfunction
