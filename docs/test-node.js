const fs = require('fs');

const wasmBuffer = fs.readFileSync('run.wasm');

let wasmInstance;
let _initialize;
let initialize;
let generate;
let decoder = new TextDecoder();

class ExitWasmError extends Error {
};

function fromCString(offset) {
    let memory = wasmInstance.exports.memory;
    let mem = new Uint8Array(memory.buffer, offset, memory.buffer.byteLength - offset);
    let length = 0;
    while (mem[length] !== 0 && length < mem.length) {
        length++;
    }
    return decoder.decode(new Uint8Array(mem.buffer, offset, length));
}

let results = [];

let wasmImports = {
    env: {
        exit: (code) => { console.log('Exit code: ', code); throw new ExitWasmError(); },
        print: (offset) => { console.log(fromCString(offset)); },
        random: (offset, size) => {
            let memory = wasmInstance.exports.memory;
            let mem = new Uint8Array(memory.buffer, offset, size);
            for (let i = 0; i < size; i++) {
                mem[i] = Math.floor((Math.random() * 256)) & 0xFF;
            }
        },
        result: (offset) => {
            results.push(fromCString(offset));
        },
    },
    wasi_snapshot_preview1: {
        fd_close: () => 0,
        fd_seek: () => -1,
        fd_write: () => -1,
    }
}

WebAssembly.instantiate(wasmBuffer, wasmImports).then(wasmModule => {
    wasmInstance = wasmModule.instance;
    ({
        memory,
        _initialize,
        initialize,
        generate,
    } = wasmInstance.exports);
    _initialize();
    initialize();
    let count = 20000;
    let t = Date.now();
    generate(20000, 1);
    t = Date.now() - t;
    console.log(results.join(''));
    console.log('Speed:', count / t * 1000, 'tokens/sec.');
});
