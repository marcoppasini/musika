![musika_logo](images/musika_logo.png)

# Musika! Fast Infinite Waveform Music Generation
Official implementation of the paper [*Musika! Fast Infinite Waveform Music Generation*](https://arxiv.org/abs/2208.08706), accepted to ISMIR 2022.  


This work was conducted as part of [Marco Pasini](https://twitter.com/marco_ppasini)'s Master thesis at the [Institute of Computational Perception](https://www.jku.at/en/institute-of-computational-perception/) at JKU Linz, with Jan Schlüter as supervisor.  

Find the __demo samples__ [here](https://marcoppasini.github.io/musika) (old 22.05 kHz implementation)  
Find the __paper__ [here](https://arxiv.org/abs/2208.08706) (old 22.05 kHz implementation)

__The current version of *musika* has been updated to produce 44.1 kHz higher quality samples!__ You can find the old implementation that is presented in the ISMIR paper in the 22kHz folder.

__Listen__ to some __44.1 kHz samples__ [here](https://www.youtube.com/watch?v=0l7OSM-bFvc)


## Online Demo
An online demo is available on Huggingface Spaces. Try it out [here](https://huggingface.co/spaces/marcop/musika)! [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/marcop/musika)

## Finetuning Colab Notebook
You can use [this Colab notebook](https://colab.research.google.com/drive/1PowSw3doBURwLE-OTCiWkO8HVbS5paRb) to train Musika on custom music, no technical skills needed!

## Installation
Before starting, make sure to have [conda](https://docs.conda.io/en/latest/miniconda.html) and [ffmpeg](https://ffmpeg.org/) installed.

First, create a new environment for *musika*:

```bash
conda create -n musika python=3.9
```

Then, activate the environment (do this every time you wish to use *musika*):

```bash
conda activate musika
```

Install CUDA 11.2 and CuDNN if you have a Nvidia GPU and you do not already have CUDA 11.2 installed on your system (you can check your CUDA version with the command *nvcc --version*):

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

And configure the system paths with the two following commands (only for Linux, skip on Windows):

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
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
python musika_test.py
```

By default the system generates *techno* samples. To generate *misc music* samples (a diverse music dataset was used for training), specify a different path for the pretrained weights:

```bash
python musika_test.py --load_path checkpoints/misc
```

You can also generate and save an arbitrary number of samples with a specified length with:

```bash
python musika_generate.py --load_path checkpoints/misc --num_samples 10 --seconds 120 --save_path generations
```

## Training
You can train a *musika* system using your own custom dataset. A pretrained encoder and decoder are provided to produce training data (in the form of compressed latent sequences) of any arbitrary domain.

> Note that using the provided universal autoencoder will sometimes produce low quality samples, for example for samples that mainly contain vocals: training a custom encoder and decoder for a specific dataset would produce higher quality samples, especially for narrow music domains. A training script for custom encoders/decoders will be provided!

Before proceeding, make sure to have a Nvidia GPU and CUDA 11.2 installed. Mixed precision is enabled by default, so if your GPU does not support it make sure to disable it using the `--mixed_precision False`  flag.

Also, you may experience errors when training using XLA (`--xla True` by default for faster training) with CUDA locally installed in the environment. See [this link](https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path) for the solution to this problem. Alternatively, specify `--xla False` (the speedup provided by XLA is not substantial).

First of all, encode audio files to training samples (in the form of compressed latent vectors) with:

```bash
python musika_encode.py --files_path folder_of_audio_files --save_path folder_of_encodings
```

`folder_of_encodings` will be automatically created if it does not exist.

*musika* encodes audio samples to chunks of sequences of latent vectors of equal length (`--max_lat_len`) which by default are double the size of the chunks used during training. During training chunks are randomly cropped as a data augmentation technique. Keep in mind that by default audio samples are required to be at least 47 s long (`--max_lat_len 512`): if you require to encode shorter samples specify a lower value (the minimum is 256, corresponding to about 23 s, which is the length of samples used for training) both during encoding and during training.

### Training from scratch
Training a model from scratch generally requires large amounts of training data if the audio domain to generate has substantial timbre diversity. However, feel free to experiment with the data you have available! The results of the system should scale quite well with the amount of data used for training. 

To train *musika* from scratch use:

```bash
python musika_train.py --train_path folder_of_encodings
```

Please be aware that training *musika* from scratch can take multiple hours on a powerful GPU (at least 2 million iterations are recommended).

A tensorboard and a gradio link will be generated for you to check losses and results during training. Additionally, generated audio samples and the corresponding spectrograms are saved in the checkpoint folder after each epoch.

To train a model with a shorter context window (6 s instead of 12 s) use:

```bash
python musika_train.py --train_path folder_of_encodings --small True
```

You can also increase the number of parameters (capacity) of the model with the `--base_channels` argument (`--base_channels 128` is the default). From our experiments, Musika can greatly benefit from an increase in capacity (at the expense of longer training times). We recommend lowering the learning rate to achieve stable training with a substantially larger model than the default one. For example:

```bash
python musika_train.py --train_path folder_of_encodings --base_channels 192 --lr 0.00007
```

Finally, if you wish to resume training from a specific checkpoint folder you can specify it in `--load_path`:

```bash
python musika_train.py --train_path folder_of_encodings --load_path checkpoints/MUSIKA_latlen_x_latdepth_x_sr_x_time_x-x/MUSIKA_iterations-xk_losses-x-x-x-x
```

### Finetuning
The fastest way to train a *musika* system on custom data is to finetune a provided checkpoint that was pretrained on a diverse music dataset. Since training *musika* from scratch usually requires large amounts of training data, finetuning can represent a good compromise in some cases. And it is incredibly fast! Note that the perfect dataset to finetune *musika* on has very limited timbre diversity (metal, piano music, ...). Finetuning on a diverse music dataset will not produce good results! __Training a *musika* system from scratch will produce better results in the majority of cases!__

You can finetune the *misc musika* model (trained on a diverse music collection) on your dataset with:

```bash
python musika_train.py --train_path folder_of_encodings --load_path checkpoints/misc --lr 0.00004
```

In case you experience nans or training instabilities try lowering the learning rate further using `--lr 0.00002`.

In case your target music domain is close to the *techno* genre, you can finetune the provided *techno* checkpoint instead of the default *misc* checkpoint:

```bash
python musika_train.py --train_path folder_of_encodings --load_path checkpoints/techno --lr 0.00004
```

Also, we provide a *misc_small* checkpoint you can finetune if you do not need to generate samples with long-range coherence (the model was trained with a shorter context window):

```bash
python musika_train.py --train_path folder_of_encodings --load_path checkpoints/misc_small --small True --lr 0.00004
```


Make sure to check out all the other flags in the *parse.py* file!

You can finally test and generate samples with your trained model with the same commands that are described in the __Generate Samples__ section above, by specifying the desired checkpoint folder in the `--load_path` argument.

## Extras

### Encode Whole Samples
By default `musika_encode.py` encodes a single audio sample to multiple training encodings of fixed length (`--max_lat_len 512` by default). If you wish to encode a single audio file to a single encoding (for use in other applications), you can specify `--whole True` in the command:

```bash
python musika_encode.py --files_path folder_of_audio_files --save_path folder_of_encodings --whole True
```

### Decode Encodings
If you wish to decode back to waveform a folder of encodings created with the `musika_encode.py` command, you can use:

```bash
python musika_decode.py --files_path folder_of_encodings --save_path folder_of_audio_files
```

`folder_of_audio_files` will be automatically created if it does not exist.

By first encoding and then decoding a collection of samples, you can check what is the upper bound on the quality of generated samples by listening to the waveform reconstructions.