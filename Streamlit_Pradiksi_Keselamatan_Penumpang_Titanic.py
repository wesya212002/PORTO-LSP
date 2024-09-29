import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
# Asumsikan titanic_df adalah dataset yang telah diolah sebelumnya
# Jika Anda ingin load dari file, gunakan pd.read_csv('file_path.csv')

titanic_df = pd.read_csv('titanic_2.csv')  # Replace with your dataset path
titanic_df = titanic_df.drop(columns = ["PassengerId", "Name", "Cabin", "Ticket"])
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'], drop_first=True)

# 2. Split data menjadi X (features) dan y (target)
X = titanic_df.drop(columns=['Survived'])  # Fitur
y = titanic_df['Survived']  # Target

# 3. Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Membuat model Logistic Regression dan melatihnya
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Membuat aplikasi Streamlit
st.title("Aplikasi Prediksi Penumpang Titanic")
st.write("""
### Gunakan model machine learning untuk memprediksi apakah penumpang Titanic selamat atau tidak.
""")

# 6. Menampilkan dataset
if st.checkbox("Tampilkan Dataset"):
    st.write(titanic_df.head())

# 7. Form input data penumpang untuk prediksi
st.sidebar.header("Input Data Penumpang")

def user_input_features():
    pclass = st.sidebar.selectbox('Kelas Penumpang (Pclass)', (1, 2, 3))
    sex = st.sidebar.selectbox('Jenis Kelamin (Sex)', ('male', 'female'))
    age = st.sidebar.slider('Umur (Age)', 0, 80, 30)
    sibsp = st.sidebar.slider('Jumlah Saudara/Istri (SibSp)', 0, 8, 1)
    parch = st.sidebar.slider('Jumlah Anak/Orang Tua (Parch)', 0, 6, 0)
    fare = st.sidebar.slider('Harga Tiket (Fare)', 0, 100, 50)
    embarked = st.sidebar.selectbox('Pelabuhan Keberangkatan (Embarked)', ('C', 'Q', 'S'))

    # Konversi input ke dataframe
    data = {'Pclass': pclass,
            'Sex': 1 if sex == 'male' else 0,  # Encoding manual untuk gender
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Menampilkan input dari user
st.write("#### Data Penumpang yang Dimasukkan:")
st.write(input_df)

# Preprocessing (jika perlu, seperti encoding atau scaling) bisa dilakukan di sini

# 8. Prediksi
prediction = model.predict(input_df)

# Menampilkan hasil prediksi
st.write("#### Hasil Prediksi:")
if prediction == 1:
    st.success("Penumpang Selamat")
else:
    st.error("Penumpang Tidak Selamat")

# Evaluasi akurasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Akurasi model Logistic Regression: {accuracy * 100:.2f}%**")
