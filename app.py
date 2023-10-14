import streamlit as st
import requests

st.title('画像分類アプリケーション')

# カテゴリ
st.sidebar.write('分類するカテゴリを1つ以上指定してください')
input1 = st.sidebar.text_input('Category_1')
input2 = st.sidebar.text_input('Category_2')
input3 = st.sidebar.text_input('Category_3')
input4 = st.sidebar.text_input('Category_4')
input5 = st.sidebar.text_input('Category_5')

# 画像ファイル（zip）
st.write('分類する画像ファイルを格納したzipファイルを指定してください')
img_file = st.file_uploader('画像ファイル（*.jpg）を圧縮したzipファイル', type='zip')

# ファイルのアップロード
def upload_file():
    num_files = 0
    extract_dir = ''
    if img_file is not None:
        files = {'file': ('uploaded_file.zip', img_file.read())}
        response = requests.post('http://127.0.0.1:8000/upload', files=files)

        extract_dir = response.json()['extract_dir']
        num_files = response.json()['num_files']

    if 0 < num_files:
        return extract_dir
    else:
        st.write('画像ファイル（*.jpg）が見つかりませんでした')
        return

# カテゴリの取得
def get_category():
    cat1 = input1.lstrip().rstrip()
    cat2 = input2.lstrip().rstrip()
    cat3 = input3.lstrip().rstrip()
    cat4 = input4.lstrip().rstrip()
    cat5 = input5.lstrip().rstrip()

    categories = []
    if 0 < len(cat1):
        categories.append(cat1)
    if 0 < len(cat2) and cat2 not in categories:
        categories.append(cat2)
    if 0 < len(cat3) and cat3 not in categories:
        categories.append(cat3)
    if 0 < len(cat4) and cat4 not in categories:
        categories.append(cat4)
    if 0 < len(cat5) and cat5 not in categories:
        categories.append(cat5)
    
    if 0 < len(categories):
        '''
        cat_other = 'その他'
        if cat_other not in categories:
            categories.append(cat_other)
        '''
        return categories
    else:
        st.write('分類するカテゴリを1つ以上指定してください')
        return

# 分類結果の表示
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
import io
def pred_disp(i, result):
      plt.figure(figsize=(4, 2))
      plt.subplot(1, 2, 1)
      #plt.title(result['image_name'])
      image = Image.open(result['image_url'])
      #image = Image.open(io.BytesIO(result['image'].encode('latin1')))
      plt.imshow(image)
      plt.axis("off")
      
      plt.subplot(1, 2, 2)
      y = np.arange(np.array(result['topk_probs']).shape[-1])
      plt.grid()
      plt.barh(y, result['topk_probs'])
      plt.gca().invert_yaxis()
      plt.gca().set_axisbelow(True)
      plt.yticks(y, [result['category'][index] for index in np.array(result['topk_categories'])])
      plt.xlim(0, 1)
      plt.xticks(np.arange(0, 1.1, 0.2))
      plt.xlabel("確率")
      
      plt.subplots_adjust(wspace=0.5)
      st.pyplot(plt)

# 分類ボタン
if st.button('分類'):
    '''
    if img_file is not None:
        files = {'file': ('uploaded_file.zip', img_file.read())}
    '''
    extract_dir = upload_file()
    categories = get_category()
    if extract_dir is not None and 0 < len(categories):
        category = {
            'categories' : categories,
            'extract_dir' : extract_dir
        }

        # 分類の実行
        response = requests.post('http://127.0.0.1:8000/categorize', json=category)

        if response.status_code == 200:
            results = response.json()['results']
        
            for i, result in enumerate(results):
                pred_disp(i, result)
        else:
            st.error('Fail')
