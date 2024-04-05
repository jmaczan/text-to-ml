# 🎮 text-to-ml

Run AutoML using natural text. Like HuggingGPT + LangChain + type inference

> 🏭 A breakdown of what is going on in code you can read on my blog: [maczan.pl](https://maczan.pl/p/lets-build-text-to-ml-an-automl-library)

It picks a right model from Hugging Face library based on user natural language query and then runs the model and parses the output to a type, inferred from the query

> ⚡ You can run this code in [Lightning AI Studio template](https://lightning.ai/jed/studios/build-your-own-automl-using-hugging-face-inference-client-and-openai-api)

<p align="center"><img width="500" src="image.png" alt="Text-to-ML"></p>

It's still an early project, so feel welcome to contribute!

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
