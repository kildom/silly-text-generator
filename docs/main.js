

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

function* tokenGenerator() {
    while (true) {
        if (results.length == 0) {
            generate(Math.min(100, sentencesRequested * 4), temperature);
        }
        yield results.shift();
    }
}

function* sentenceGeneratorCreate() {
    let gen = tokenGenerator();
    let value = gen.next().value;
    while (true) {
        let sentence = value;
        while (value.indexOf('.') < 0) {
            value = gen.next().value;
            sentence += value;
        }
        value = gen.next().value;
        while (!value.match(/^([\w\s]+$|-\w|~)/)) {
            sentence += value;
            value = gen.next().value;
        }
        yield sentence;
    }
}

function S(q) {
    return document.querySelector(q);
}

let sentencesRequested = 0;
let timer = null;
let temperature = 1;
let lastParagraph = null;
let paragraphMin = 2;
let paragraphMax = 15;
let paragraphScale = 1;
let paragraphRemaining = 1;
let sentenceGenerator = null;

function generateChunk() {
    timer = null;
    let done = 0;
    while (sentencesRequested > 0 && done < Math.min(20, Math.max(5, sentencesRequested / 10))) {
        if (lastParagraph == null || paragraphRemaining <= 0) {
            lastParagraph = document.createElement('p');
            lastParagraph.className = 'output';
            S('#content').appendChild(lastParagraph);
            if (paragraphScale < 0) {
                paragraphRemaining = Number.POSITIVE_INFINITY;
            } else {
                paragraphRemaining = Math.ceil((paragraphMin + Math.random() * (paragraphMax + 1 - paragraphMin)) * paragraphScale);
            }
        }
        lastParagraph.innerHTML += sentenceGenerator.next().value;
        sentencesRequested--;
        paragraphRemaining--;
        done++;
    }
    S('#numLeft').innerHTML = sentencesRequested;
    if (sentencesRequested == 0) {
        S('#numLeft').style.display = 'none';
    } else {
        S('#numLeft').style.display = 'inline';
    }
    startGenerating(1);
}

function startGenerating(ms) {
    if (sentencesRequested > 0 && timer === null) {
        timer = setTimeout(generateChunk, ms || 1);
    }
}

function generateClick(event) {
    let type = event.srcElement.getAttribute("data-type");
    if (type == 'clear') {
        S('#content').innerHTML = '';
        sentencesRequested = 0;
        paragraphRemaining = 0;
        lastParagraph = null;
    } else if (Number.isInteger(parseInt(type))) {
        sentencesRequested += parseInt(type);
        startGenerating();
    }
}

function temperatureClick(event) {
    let temp = parseFloat(event.srcElement.getAttribute("data-value"));
    if (!Number.isNaN(temp)) {
        temperature = temp;
        document.querySelectorAll('#temperatureButtons>div').forEach(e => e.setAttribute("data-sel", "0"));
        event.srcElement.setAttribute("data-sel", "1");
    }
}

function sizeClick(event) {
    let s = parseFloat(event.srcElement.getAttribute("data-value"));
    if (!Number.isNaN(temp)) {
        paragraphScale = s;
        if (paragraphRemaining == Number.POSITIVE_INFINITY) {
            paragraphRemaining = 1;
        }
        document.querySelectorAll('#sizeButtons>div').forEach(e => e.setAttribute("data-sel", "0"));
        event.srcElement.setAttribute("data-sel", "1");
    }
}

function wasmReady() {
    /*let count = 20000;
    let t = Date.now();
    generate(20000, 1);
    t = Date.now() - t;
    console.log(results.join(''));
    console.log('Speed:', count / t * 1000, 'tokens/sec.');*/
    sentenceGenerator = sentenceGeneratorCreate();
    S('#wait').style.display = 'none';
    S('#main').style.display = '';
    S('#generateButtons').addEventListener('click', generateClick);
    S('#temperatureButtons').addEventListener('click', temperatureClick);
    S('#sizeButtons').addEventListener('click', sizeClick);
}

function wasmLoaded() {
    _initialize();
    initialize();
    wasmReady();
}

async function downloadWasm(url) {

    let wasmModule = await WebAssembly.instantiateStreaming(fetch(url), wasmImports);
    wasmInstance = wasmModule.instance;
    ({
        memory,
        _initialize,
        initialize,
        generate,
    } = wasmInstance.exports);
}

window.__my__load_wasm = async function (dataURL) {
    await downloadWasm(dataURL);
    wasmLoaded();
}

window.onload = async function () {
    try {
        await downloadWasm('run.wasm');
    } catch (e) {
        let s = document.createElement('script');
        s.src = 'run.wasm.js';
        document.body.appendChild(s);
        return;
    }
    wasmLoaded();
}
