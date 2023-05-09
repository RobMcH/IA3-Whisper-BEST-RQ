# Training Whisper-like Models using IA^3 and BEST-RQ

This repository contains an implementation of Google's BERT-based Speech pre-Training
with Random-projection Quantizer (BEST-RQ) ([Arxiv](https://arxiv.org/abs/2202.01855)) pre-training objective. Both the
original method as well as the more recent adaption of BEST-RQ in the Universal Speech Model
([Arxiv](https://arxiv.org/abs/2303.01037)) are implemented.

Further, the repository extends the OpenAI Whisper repository by implementing an IA^3
([Arxiv](https://arxiv.org/abs/2205.05638)) trainable Whisper implementation. At the moment, IA^3 is only implemented
for the AudioEncoder.

The repository is currently work in progress with coming features still to be implemented.

## Installation

An easy-to-use Dockerfile is provided. After cloning the repository, the Docker image can be built using
```bash
docker build -t best_rq .
```
and a corresponding container can be started using
```bash
docker run -e <WANDB_API_KEY> --gpus all -it best_rq
```
where `<WANDB_API_KEY>` should be replaced with a Weights and Biases API key. If a valid API key is provided, metrics
and (intermediate) checkpoints are logged automatically to Weights and Biases. As IA^3 requires only a tiny amount of
trainable parameters, checkpoints are (individually) negligible in space requirements.

The Dockerfile contains a micromamba installation, so further libraries/requirements can easily be installed on the fly.
As a GPU is highly recommended for running the training, the image is only tested on a machine with an installed GPU.

## Training

IA^3-based training of a (pretrained) Whisper model using BEST-RQ can be launched using the `train` command within a
Docker container. E.g., a training using a batch size of `4`, `16` BEST-RQ codebooks, and gradient accumulation over
`256` batches can be launched like this:
```bash
train --batch_size 4 --num_codebooks 16 --accumulate_gradients 256
```

The full list of possible hyper-parameters can be obtained via
```bash
train --help
```
