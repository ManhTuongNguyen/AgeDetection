import random
import string


def get_random_string(length: int = 10) -> str:
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string
