import os
import requests
from bs4 import BeautifulSoup

url = 'http://www.agritech.tnau.ac.in/expert_system/paddy/Botany.html'

response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

text = soup.get_text()

print(" ".join(text.split()))