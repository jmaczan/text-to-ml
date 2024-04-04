import requests
import os
from dotenv import load_dotenv
from PIL import Image
import io

from api import endpoint

load_dotenv()


def ai(query, data=None):
    payload = {"text": query}

    files = {}

    if isinstance(data, str) and (
        data.startswith("http://") or data.startswith("https://")
    ):
        response = requests.get(data)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            files["file"] = (
                "image.png",
                buf,
                "image/png",
            )
        else:
            return
    elif isinstance(data, str):
        payload["payload_text"] = data
    elif isinstance(data, (int, float)):
        payload["payload_text"] = str(data)
    elif isinstance(data, Image.Image):
        buf = io.BytesIO()

        image_format = data.format if data.format else "PNG"

        format_extension = image_format.lower()
        if format_extension == "jpeg":
            format_extension = "jpg"
        mime_type = f"image/{format_extension}"

        data.save(buf, format=image_format)
        buf.seek(0)

        files["file"] = (
            f"image.{format_extension}",
            buf,
            mime_type,
        )
    elif data is not None:
        if os.path.isfile(data):
            mime_type = "application/octet-stream"
            if data.endswith(".mp3"):
                mime_type = "audio/mpeg"
            elif data.endswith(".mp4"):
                mime_type = "video/mp4"
            elif data.endswith(".wav"):
                mime_type = "audio/wav"

            files["file"] = (os.path.basename(data), open(data, "rb"), mime_type)
        else:
            return
    response = requests.post(
        endpoint(os.environ["API_URL"], "run"),
        data=payload,
        files=files,
    )

    response_json = response.json()

    if response_json == "true":
        return True
    if response_json == "false":
        return False

    try:
        response_as_int = int(response_json)
        return response_as_int
    except ValueError as e:
        noop()

    return response_json


def noop():
    return None
