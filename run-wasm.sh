#!/bin/bash

/home/doki/my/ZeroTierOne-1.10.4/tools/wasi-sdk-19.0/bin/clang \
    --sysroot /home/doki/my/ZeroTierOne-1.10.4/tools/wasi-sdk-19.0/share/wasi-sysroot/ \
    -mexec-model=reactor \
    -O3 \
    -o pages/run.wasm \
    run-wasm.c -lm

echo -n "window.__my__load_wasm('data:application/wasm;base64," > pages/run.wasm.js
base64 -w 0 pages/run.wasm >> pages/run.wasm.js
echo "');" >> pages/run.wasm.js
