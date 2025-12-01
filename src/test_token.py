import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
print("Token loaded:", token[:10] + "********")
