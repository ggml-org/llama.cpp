::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT

set INPUT2="Building a website can be done in 10 simple steps:\nStep 1:"
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force


.\build\bin\llama-cli.exe -m models\llama-2-7b.Q4_0.gguf -p %INPUT2% -n 400 -e -ngl 33 -s 0
