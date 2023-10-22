import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from model import predict

# スタイリングとデザインの変更
st.markdown("""
<style>
body {
    color: #333;
    background-color: #f4f4f4;
}
</style>
    """, unsafe_allow_html=True)

# メインエリアの配置の変更
st.title("画像認識アプリ")
st.write("オリジナルの画像認識モデルを使って何の画像かを判定します。")

# サイドバーに画像のソース選択を配置
img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
elif img_source == "カメラで撮影":
    img_file = st.sidebar.camera_input("カメラで撮影")

# 画像の前処理のためのスライダー
brightness = st.sidebar.slider("明るさ調整", 0.5, 1.5, 1.0)
contrast = st.sidebar.slider("コントラスト調整", 0.5, 1.5, 1.0)

if img_file is not None:
    # 中央にアップロードした画像を表示
    img = Image.open(img_file)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    st.image(img, caption="対象の画像", use_column_width=True)

    with st.spinner("推定中..."):
        # 予測
        results = predict(img)

        # 結果の表示
        st.subheader("判定結果")
        n_top = 3  # 確率が高い順に3位まで返す
        for result in results[:n_top]:
            st.write(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")

        # 円グラフの表示
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")  # その他
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(1.0 - sum(pie_probs))  # その他
        fig, ax = plt.subplots()
        wedgeprops={"width": 0.3, "edgecolor": "white"}
        textprops = {"fontsize": 10}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f%%", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig)

st.sidebar.caption("""
このアプリは、「Fashion-MNIST」を訓練データとして使っています。\n
Copyright (c) 2017 Zalando SE\n
Released under the MIT license\n
https://github.com/zalandoresearch/fashion-mnist#license
""")
