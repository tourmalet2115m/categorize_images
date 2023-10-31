from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import zipfile
import os
import shutil
from datetime import datetime

# googletransをインポートする。
#import locale
#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
from googletrans import Translator

# CLIPモデルの読み込み
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import io
import base64

model_name = 'openai/clip-vit-base-patch16'
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

# 分類条件
from pydantic import BaseModel
class category(BaseModel):
    categories: list
    extract_dir: str

class ImageInfo(BaseModel):
    file_name: str
    file_url: str

# 英語から日本語に翻訳
from googletrans import Translator
def transJp2En(text_ja: str):
    translator = Translator()
    text_en = translator.translate(text_ja, src='ja', dest='en').text
    return text_en

# 画像ファイルを分類
import glob
import time
def categorize_images(category: category):
    # 画像ファイルを取得
    '''
    extract_dir = category.extract_dir
    files = glob.glob(extract_dir + '/*.jpg')
    files.sort()
    images = [Image.open(file) for file in files]
    '''
    start_time = time.time()
    extract_dir = category.extract_dir
    images = []
    image_urls = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            file_url = os.path.join(root, file)
            images.append(Image.open(file_url))
            image_urls.append(ImageInfo(file_name=file, file_url=file_url))
    print(f'画像ファイル取得：{time.time() - start_time}秒')

    # カテゴリを取得
    start_time = time.time()
    cat_list = category.categories
    labels = []
    for cat in cat_list:
        cat = cat.lstrip().rstrip()
        if len(cat) == 0:
            continue
        cat_en = transJp2En(cat)
        if cat_en in labels:
            continue
        labels.append(cat_en)
    label_descriptions = [f"This is a photo of a {label}" for label in labels]
    print(f'カテゴリ取得：{time.time() - start_time}秒')

    # 画像ファイルとカテゴリをtensorに変換
    start_time = time.time()
    inputs = processor(text=label_descriptions, 
                       images=images, return_tensors="pt", padding=True).to(device)
    print(f'tensor変換：{time.time() - start_time}秒')

    # 画像ファイルごとのCOS類似度、確率を計算
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(1)
    print(f'確率算出：{time.time() - start_time}秒')
    
    start_time = time.time()
    image_results = []
    for i, image_url in enumerate(image_urls):
        image_probs = probs[i]
        topk_probs, topk_categories = image_probs.topk(len(labels))
        '''
        image_bin = io.BytesIO()
        image.save(image_bin, format='JPEG')
        image_bin.seek(0)
        image_data = image_bin.read()
        '''
        image_result = {
            #'image': image_data.decode('latin1'),
            'image_url': image_url.file_url,
            'image_name': image_url.file_name,
            'category': cat_list,
            'topk_categories': topk_categories.tolist(),
            'topk_probs': topk_probs.tolist()
        }
        image_results.append(image_result)
    print(f'response作成：{time.time() - start_time}秒')

    return JSONResponse(content={'results': image_results})

app = FastAPI()

# トップページ
@app.get('/')
def index():
    return {'Categorize Images'}

# zipファイルアップロード
@app.post('/upload')
async def upload(file: UploadFile):
    extract_dir = 'extracted_files\\' + datetime.now().strftime('%Y%m%d%H%M%S%f')
    extract_dir = extract_dir[:-3]

    filename = file.filename
    with open(filename, 'wb') as f:
        f.write(file.file.read())

    extracted_files = []
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            os.makedirs(extract_dir, exist_ok=True)
            zip_ref.extractall(extract_dir)
            extracted_files = os.listdir(extract_dir)

    return JSONResponse(content={'num_files': len(extracted_files), 'extract_dir': extract_dir})

# ファイル削除
@app.delete('/delete/{dir}')
async def deleteDir(dir: str):
    if os.path.exists(dir):
        shutil.rmtree(dir)

# 画像分類
@app.post('/categorize')
async def categorize(category: category):
    return categorize_images(category)
