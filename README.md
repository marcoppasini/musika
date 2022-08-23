![musika_logo](images/musika_logo.png)

# Musika! Fast Infinite Waveform Music Generation
Official implementation of the paper [*Musika! Fast Infinite Waveform Music Generation*](https://arxiv.org/abs/2208.08706), accepted at ISMIR 2022.  


This work was conducted as part of [Marco Pasini](https://twitter.com/marco_ppasini)'s Master thesis at the [Institute of Computational Perception](https://www.jku.at/en/institute-of-computational-perception/) at JKU Linz, with Jan Schlüter as supervisor.  

Find the __demo samples__ [here](https://marcoppasini.github.io/musika)  
Find the __paper__ [here](https://arxiv.org/abs/2208.08706)

## Online Demo
An online demo is available on [Hugging Face Spaces](https://huggingface.co/spaces). Try it out [here](https://huggingface.co/spaces/marcop/musika)! [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/marcop/musika)

## The code will be available soon!

<!-- ## Installation
First of all, make sure to have [conda](https://www.anaconda.com/products/distribution) and [ffmpeg](https://ffmpeg.org/) installed.

First, create a new environment for *musika*:

```bash
conda create -n musika python=3.9
```

Then, activate the environment (do this every time you wish to use *musika*):

```bash
conda activate musika
```

Finally, clone this repository, move to its directory and install the requirements:

```bash
git clone https://github.com/marcoppasini/musika
cd musika
pip install -r requirements.txt
```

## Generate Samples
You can conveniently generate samples using a [Gradio](https://gradio.app/) interface by running the command:

```bash
python3 musika.py
```

By default the system generates *classical music* samples. To generate *techno* samples, specify a different path for the pretrained weights:

```bash
python3 musika.py --load_path checkpoints/techno
```

## Training
You can train a *musika* system using your own custom dataset. A pretrained encoder and decoder are provided to produce training data (in the form of compressed latent sequences) of any arbitrary domain.

> Please note that using the provided universal encoder will produce lower quality samples: training a custom encoder and decoder for a specific dataset would produce higher quality samples, especially for narrow music domains. A training script for custom encoders/decoders will be provided in the future!

Before proceeding, make sure to have a GPU with *cuda* installed. Mixed precision is enabled by default, so if your GPU does not support it make sure to disable it using the `--mixed_precision False`  flag.

First of all, encode audio files to training samples with:

```bash
python3 musika_encode.py --files_path folder_of_audio_files/ --save_path output_folder/
```

Then, you can train a custom musika system using:

```bash
python3 musika_train.py --train_path output_folder/
```

Make sure to check out all the other flags in the *parse.py* file!

 -->
