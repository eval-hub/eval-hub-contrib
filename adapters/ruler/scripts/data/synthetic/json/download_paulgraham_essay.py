# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import shutil
import glob
import json
import urllib.request
import html2text
from bs4 import BeautifulSoup
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
URLS_FILE = os.path.join(DATA_DIR, "PaulGrahamEssays_URLs.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "PaulGrahamEssays.json")
TEMP_REPO_DIR = os.path.join(DATA_DIR, "essay_repo")
TEMP_HTML_DIR = os.path.join(DATA_DIR, "essay_html")

os.makedirs(TEMP_REPO_DIR, exist_ok=True)
os.makedirs(TEMP_HTML_DIR, exist_ok=True)

h = html2text.HTML2Text()
h.ignore_images = True
h.ignore_tables = True
h.escape_all = True
h.reference_links = False
h.mark_code = False

with open(URLS_FILE) as f:
    urls = [line.strip() for line in f]

for url in tqdm(urls):
    if '.html' in url:
        filename = url.split('/')[-1].replace('.html', '.txt')        
        try:
            with urllib.request.urlopen(url) as website:
                content = website.read().decode("unicode_escape", "utf-8")
                soup = BeautifulSoup(content, 'html.parser')
                specific_tag = soup.find('font')
                parsed = h.handle(str(specific_tag))
                
                with open(os.path.join(TEMP_HTML_DIR, filename), 'w') as file:
                    file.write(parsed)
        
        except Exception as e:
            print(f"Fail download {filename}, ({e})")

    else:
        filename = url.split('/')[-1]
        try:
            with urllib.request.urlopen(url) as website:
                content = website.read().decode('utf-8')
            
            with open(os.path.join(TEMP_REPO_DIR, filename), 'w') as file:
                file.write(content)
                    
        except Exception as e:
            print(f"Fail download {filename}, ({e})")

files_repo = sorted(glob.glob(os.path.join(TEMP_REPO_DIR, '*.txt')))
files_html = sorted(glob.glob(os.path.join(TEMP_HTML_DIR, '*.txt')))
print(f'Download {len(files_repo)} essays from `https://github.com/gkamradt/LLMTest_NeedleInAHaystack/`') 
print(f'Download {len(files_html)} essays from `http://www.paulgraham.com/`') 

text = ""
for file in files_repo + files_html:
    with open(file, 'r') as f:
        text += f.read()
        
if not files_repo and not files_html:
    raise RuntimeError("No Paul Graham essays were downloaded")

with open(OUTPUT_FILE, 'w') as f:
    json.dump({"text": text}, f)


shutil.rmtree(TEMP_REPO_DIR)
shutil.rmtree(TEMP_HTML_DIR)
