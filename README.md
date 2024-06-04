# 🎮 text-to-ml

> 💜 PagedOut #4 Issue ["Building automated machine learning with type inference"](https://pagedout.institute/download/PagedOut_004_beta1.pdf#page=4)

Run AutoML using natural text. Like HuggingGPT + LangChain + type inference

> 🏭 A breakdown of what is going on in code you can read on my blog: [maczan.pl](https://maczan.pl/p/lets-build-text-to-ml-an-automl-library)

It picks a right model from Hugging Face library based on user natural language query and then runs the model and parses the output to a type, inferred from the query

<p align="center"><img width="500" src="https://github.com/jmaczan/text-to-ml/assets/18054202/63367fd9-5db9-46a2-8ec7-e17f5c8e2863" alt="Text-to-ML"></p>

> ⚡ You can run this code in [Lightning AI Studio template](https://lightning.ai/jed/studios/build-your-own-automl-using-hugging-face-inference-client-and-openai-api)
<p>
<a target="_blank" href="https://lightning.ai/jed/studios/build-your-own-automl-using-hugging-face-inference-client-and-openai-api">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>
</p>

It's still an early project and **you are welcome to contribute**!

## Setup

1. Get OpenAI API Key
2. Get Hugging Face API Key
3. Create an assistant and copy its id
4. Create `.env` file and fill it with values:

```
OPENAI_API_KEY=
HF_TOKEN=
```

## Build

```sh
conda create -n text-to-ml python=3.9
conda activate text-to-ml
conda install --file requirements.txt
```

## Run

```
python app.py
```

## Run experiments

```
python experiments.py
```

## License

GPLv3

## Author

Jędrzej Paweł Maczan, Poland, 2024
