# 1) Set program arguments
set args -m ~/dev/llm/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF/deepseek-r1-distill-qwen-1.5b-q4_0.gguf -b 16 -ngl 0 -c 1024 -t 4 -p "Hello"

# 2) Redirect GDB output to a log file
set logging file gdb_output.log
set logging on

# 3) Place a breakpoint at the debug_hook() function in ggml-cpu.c
break ggml-cpu.c:debug_hook

# 4) Commands to execute once the breakpoint is hit
commands
  # Prevent GDB from printing its usual breakpoint messages
  silent

  # (a) Exit from debug_hook() and return to its caller
  #     This should land you at check_invalid_values() right before 'return true;'
  finish

  # (b) Now that you're in check_invalid_values(), print variables of interest
  p *src0
  p (*src0).data
  x/128f (*src0).data

  # (c) If you only want to trigger once, disable the breakpoint afterwards
  disable $bpnum

  # If you would rather keep hitting this breakpoint repeatedly, comment out
  # the disable command above and uncomment the following 'continue' command:
  # continue
end

# 5) Automatically run the program (remove or comment out if you want to run manually)
run
