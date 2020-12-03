from PIL import Image #PillowからImageクラスをインポート
import os, glob #OSにアクセスする関数をインポート
import numpy as np #NumPyをインポート
from sklearn import model_selection #トレーニングとテストデータを分割する関数

# リスト型変数に入れる
classes = ["car","ship","airplane"]
# リストサイズを取得
num_classes = len(classes)
# 画像データの変換サイズ。縦横ピクセル数
image_size = 50 

X = [] #リスト型変数Xを初期化
Y = [] #リスト型変数Yを初期化


for index, classlabel in enumerate(classes): # classesから値を取り出し付番
    # ディレクトリ名を生成
    photos_dir = "./" + classlabel
    
    # ファイル一覧を取得
    files = glob.glob(photos_dir + "/*.jpg")
    
    # 各ファイルをNumPyアレーに変換し、リストに追加
    for i, file in enumerate(files): 
        #filesから1個づつ取り出しfileに入れ付番
        
        if i >= 200: break #200を超えたら次のラベルのループへ
        image = Image.open(file) #Imageクラスのopen関数でファイルをオープン
        image = image.convert("RGB") #配色データをRGBの順に揃える
        image = image.resize((image_size, image_size)) #画像サイズを揃える
        data = np.asarray(image) #NumPyアレーに変換
        X.append(data) #リストXの末尾に追加
        Y.append(index) #リストYの末尾に追加

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)

xy = (X_train, X_test, y_train, y_test)

np.save("./vehicl.npy", xy)