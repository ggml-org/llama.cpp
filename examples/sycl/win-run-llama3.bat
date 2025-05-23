::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT

set INPUT2="Building a website can be done in 10 simple steps:\nStep 1:"
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force


.\build\bin\llama-cli.exe -m models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -p %INPUT2% -n 400 -e -ngl 99
