import json
from fastapi.exceptions import RequestValidationError
import uvicorn
from dotenv import load_dotenv
import os
from openai import OpenAI
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi import (
    FastAPI,
    Form,
    Request,
    UploadFile,
    Response,
    status,
    File,
)
import logging
import httpx
from pydantic import BaseModel
from typing import List
from huggingface_hub import login, InferenceClient

app = FastAPI()

limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
timeout = httpx.Timeout(timeout=5.0, read=15.0)
client = httpx.AsyncClient(limits=limits, timeout=timeout)


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
private_api_key = os.environ["SECRET"]

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class Item(BaseModel):
    query: str
    text: Optional[str] = None


class ItemList(BaseModel):
    items: List[Item]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):

    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


@app.post("/run")
async def run(
    response: Response,
    text: str = Form(...),
    file: List[UploadFile] = File(...),
):

    try:
        task_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": text}],
            tools=tools_run,
        )

        if "answer" in json.loads(
            task_response.choices[0].message.tool_calls[0].function.arguments
        ):
            return json.loads(
                task_response.choices[0].message.tool_calls[0].function.arguments
            )["answer"]

        task_type = json.loads(
            task_response.choices[0].message.tool_calls[0].function.arguments
        )["task_type"]

        return_type = json.loads(
            task_response.choices[0].message.tool_calls[0].function.arguments
        )["return_type"]

        login(token=os.environ["HF_TOKEN"])

        # TODO: I can extract this to a separate, plug-in module so people can have their own models registries, alternative to Hugging Face, as long as it follows the same API structure
        inference_client = InferenceClient()

        method_to_call = getattr(
            inference_client, task_type, None
        )  # TODO: If outcome is not satisfactory, I can try with more than a single task_type, like second_task_type

        if callable(method_to_call):
            call_key = data_to_inference_client_call_property(task_type)
            if call_key in ["audio", "image"]:
                file = await file[0].read()
            result = method_to_call(
                **{
                    data_to_inference_client_call_property(task_type): file,
                }
            )

        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "user",
                    "content": f"Given an API response to the user's request, synthesize a response to the original user's request taking the API response into account in ordert to respond with a type, specified below. Don't include anything in your response, besides a single most important fact from response in a specified type. If possible, respond with a single word. If not, then limit amount of words as much as possible, to not include any word that is not 100% necessary to provide the response. If a question is about numbers, respond with a numeric value only. API response: '{result}'. User's request: '{text}''. Response type: '{return_type}'",
                }
            ],
        )

        return final_response.choices[0].message.content.lower()

    except Exception as err:
        return {"error": "An error occured"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=(8080 if os.environ.get("PORT") is None else int(os.environ.get("PORT"))),
        reload=True,
    )


def extract_relevant_info(output):
    # Audio Classification
    if isinstance(output, list) and all(
        isinstance(el, dict) and "score" in el for el in output
    ):
        return max(output, key=lambda x: x["score"])["label"]

    # Audio to Audio
    elif isinstance(output, list) and all(
        isinstance(el, dict) and "blob" in el for el in output
    ):
        return [el["blob"] for el in output]

    # Automatic Speech Recognition
    elif isinstance(output, dict) and "text" in output:
        return output["text"]

    # Chat Completion or Conversational
    elif isinstance(output, dict) and "generated_text" in output:
        return output["generated_text"]

    # Document Question Answering
    elif isinstance(output, list) and all(
        isinstance(el, dict) and "answer" in el for el in output
    ):
        return max(output, key=lambda x: x["score"])["answer"]

    # Feature Extraction
    elif isinstance(output, list) and all(isinstance(el, list) for el in output):
        return output  # List of feature vectors

    # Fill Mask
    elif isinstance(output, list) and all(
        isinstance(el, dict) and "sequence" in el for el in output
    ):
        return max(output, key=lambda x: x["score"])["sequence"]

    # Image Classification
    elif isinstance(output, list) and all(
        isinstance(el, dict) and "label" in el for el in output
    ):
        return max(output, key=lambda x: x["score"])["label"]

    # Image Segmentation
    elif isinstance(output, list) and all(
        isinstance(el, dict) and "mask" in el for el in output
    ):
        return [el["mask"] for el in output]  # List of masks

    # Image to Image
    elif "blob" in output:
        return output["blob"]

    elif isinstance(output, str):
        return output

    else:
        return "Output format not recognized"


def data_to_inference_client_call_property(task_type):
    task_to_field = {
        "audio_classification": "audio",
        "audio_to_audio": "audio",
        "automatic_speech_recognition": "audio",
        "text_to_speech": "text",
        "image_classification": "image",
        "image_segmentation": "image",
        "image_to_image": "image",
        "image_to_text": "image",
        "object_detection": "image",
        "text_to_image": "text",
        "zero_shot_image_classification": "image",
        "document_question_answering": "document",
        "visual_question_answering": "image",
        "conversational": "text",
        "feature_extraction": "text",
        "fill_mask": "text",
        "question_answering": "question",
        "sentence_similarity": "text",
        "summarization": "text",
        "table_question_answering": "table",
        "text_classification": "text",
        "text_generation": "inputs",
        "token_classification": "text",
        "translation": "text",
        "zero_shot_classification": "text",
        "tabular_classification": "data",
        "tabular_regression": "data",
    }

    field = task_to_field.get(task_type, None)

    if field is not None:
        return field
    else:
        raise ValueError("Unsupported task type")


tools_run = [
    {
        "type": "function",
        "function": {
            "name": "tools_run",
            "description": "User writes what they want to know or achieve or ask about. Choose what type of task should machine learning model perform, based on user's description of a model or user's description of what user expects from a model. Choose what type of return value should it be, based on user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": [
                            "audio_classification",
                            "audio_to_audio",
                            "automatic_speech_recognition",
                            "text_to_speech",
                            "image_classification",
                            "image_segmentation",
                            "image_to_image",
                            "image_to_text",
                            "object_detection",
                            "text_to_image",
                            "zero_shot_image_classification",
                            "document_question_answering",
                            "visual_question_answering",
                            "conversational",
                            "feature_extraction",
                            "fill_mask",
                            "question_answering",
                            "sentence_similarity",
                            "summarization",
                            "table_question_answering",
                            "text_classification",
                            "text_generation",
                            "token_classification",
                            "translation",
                            "zero_shot_classification",
                            "tabular_classification",
                            "tabular_regression",
                        ],
                        "description": "Type of task should machine learning model perform, based on user's description of a model or user's description of what user expects from a model.",
                    },
                    "return_type": {
                        "type": "string",
                        "enum": ["boolean", "string", "number", "image", "audio"],
                        "description": "Decide what type of value corresponds the most to the output from the user's query.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "If you are confident that able to fulfill user request, so it's not needed to create a ML model that is able to fulfill the request, put your answer into field 'answer'. If you're not sure, leave this field empty.",
                    },
                },
                "required": ["task_type", "return_type"],
            },
        },
    },
]
