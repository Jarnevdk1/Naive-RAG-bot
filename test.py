from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()  # reads GOOGLE_API_KEY from your environment

for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)