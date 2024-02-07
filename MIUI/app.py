from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Membaca data dari file Excel
data_path = "DATA.xlsx"
df = pd.read_excel(data_path)

# Pisahkan fitur dan label
X = df['Komen']
y = df['Label']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bangun dan latih model SVM
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', C=1))
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        link_input = request.form['link']
        link_data = [link_input]

        prediction = model.predict(link_data)
        print("Nilai prediksi:", prediction[0])  # Tambahkan pernyataan cetak
        result = prediction[0]
        print("Hasil prediksi:", prediction[0])  # Tambahkan pernyataan cetak
        return render_template('index.html', link=link_input, result=result)

if __name__ == '__main__':
    app.run(debug=True)
