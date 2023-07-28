# Silly text generator

See the demo on https://kildom.github.io/silly-text-generator/

My goal was to create a small and fast generator that runs in a browser and can be used as a placeholder text generator for the web.
I based this generator on the [llama2.c](https://github.com/karpathy/llama2.c) project. It was trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Because of its size, the model is not able to create stories that make any sense, but they look like normal text.

That's just a proof-of-concept project. The code is really messy and far from
high quality.
