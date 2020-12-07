from keras.models import Sequential, load_model #kerasのニューラルネットワーククラス, パラメータ読み込み関数
from keras.layers import Conv2D, MaxPooling2D #ニューラルネットワークのレイヤーを生成する関数
from keras.layers import Activation, Dropout, Flatten, Dense #活性化関数, ドロップアウト, 直列化, 全結合のレイヤーを生成する関数
import keras,sys #Kerasとシステムにアクセスする関数
import numpy as np #NumPy
from PIL import Image #Pillowに含まれるImageクラス

# クラスラベルリスト
classes = ["car", "airplane", "ship"]
num_classes = len(classes) #クラス数
image_size = 50 #画像サイズ

def build_model(): #モデルを定義する関数
    # モデルをロードし、インスタンスに格納
    model = load_model('./vehicl_cnn.h5')

    # モデルのインスタンスを返す
    return model

def main():
    image = Image.open(sys.argv[1]) #ファイルのオープン
    image = image.convert('RGB') #RGBへのカラーチャンネルの変更
    image = image.resize((image_size, image_size)) #リサイズ
    data = np.asarray(image)/255 #正規化処理
    X = [] #空のリストを生成, 初期化
    X.append(data) #NumPy配列をリスト末尾に追加
    X = np.array(X) #XをNumPy配列に変換
    #モデルを生成する
    model = build_model()
    result = model.predict([X])[0] #Xをpredict関数に渡し、先頭のデータを取り出す
    predicted = result.argmax() #推定確率が最大になる配列添字を求める
    percentage = int(result[predicted] * 100) #最大確率のスコアをパーセント表記に
    print("{0} ({1} %)".format(classes[predicted], percentage))

if __name__ == "__main__":
    main()