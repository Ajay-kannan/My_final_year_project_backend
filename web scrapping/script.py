import os
import requests
from bs4 import BeautifulSoup

# Function to scrape data from HTML content
def scrape_html_content(url):
    # Send a GET request to the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    results = soup.find("body")
    atag = results.find_all("a")
    alist = []
    for a in atag:
        alist.append(str(a['href']))
   
    for index, filename in enumerate(alist):
        print(filename)
        if filename.endswith('.html') or filename.endswith('.htm'):
            url = html_url + filename
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            
            text_final = " ".join(text.split())
            with open(f"ragi.txt", 'a+' , encoding='utf-8') as file:
                file.write(text_final + '\n')

        else:
            print(f"Failed to fetch data from URL: {filename}")

# URL of the HTML page
html_url = 'http://www.agritech.tnau.ac.in/expert_system/ragi/'

# Call the function to scrape HTML content
scrape_html_content(html_url)
