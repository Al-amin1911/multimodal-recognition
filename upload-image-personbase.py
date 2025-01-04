# Import the SDK
from opencv.fr import FR
from opencv.fr.search.schemas import SearchRequest, VerificationRequest
from opencv.fr.persons.schemas import PersonBase

BACKEND_URL = "https://eu.opencv.fr"
DEVELOPER_KEY = ""

# Initialize the SDK
sdk = FR(BACKEND_URL, DEVELOPER_KEY)

person = PersonBase(
    [
        "img_file_path",
        "img_file_path",
        "img_file_path"
    ],
    name="person_name",
)

person = sdk.persons.create(person)
