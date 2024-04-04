from ai import ai
from random import choice
from PIL import Image


if __name__ == "__main__":
    image = Image.open("test_data/dog.jpg")

    print(ai("is there a dog on the image?", image))
    print(ai("what dog race is on the image?", image))
