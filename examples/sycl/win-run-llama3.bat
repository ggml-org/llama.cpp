::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT

set INPUT2="Building a website can be done in 10 simple steps:\nStep 1:"
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force

:: support malloc device memory more than 4GB.
set UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1

.\build\bin\llama-completion.exe -m models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -no-cnv -p %INPUT2% -n 400 -s 0 -e -ngl 99
