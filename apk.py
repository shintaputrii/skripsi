import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numpy import array
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Prediksi Kualitas Udara DKI Jakarta",
    page_icon="https://raw.githubusercontent.com/shintaputrii/skripsi/main/house_1152964.png",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

st.write(
    """<h1> Prediksi Kualitas Udara di DKI Jakarta</h1>""",
    unsafe_allow_html=True,
)

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write(
                """<h2 style = "text-align: center;"><img src="https://raw.githubusercontent.com/shintaputrii/skripsi/main/house_1152964.png" width="130" height="130"><br></h2>""",
                unsafe_allow_html=True,
            ),
            [
                "Home",
                "Data",
                "Missing Value",
                "Modeling",
                "Next Day",

            ],
            icons=[
                "house",
                "file-earmark-font",
                "bar-chart",
                "gear",
                "arrow-down-square",
                "person",
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#005980"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "color": "white",
                },
                "nav-link-selected": {"background-color": "#005980"},
            },
        )

    if selected == "Home":
        st.write(
            """<h3 style = "text-align: center;">
        <img src="https://raw.githubusercontent.com/shintaputrii/skripsi/main/udara.jpeg" width="500" height="300">
        </h3>""",
            unsafe_allow_html=True,
        )

        st.subheader("""Deskripsi Aplikasi""")
        st.write(
            """
         Aplikasi Prediksi kualitas Udara di DKI Jakarta merupakan aplikasi yang digunakan untuk meramalkan lima konsentrasi polutan udara di DKI Jakarta yang meliputi PM10, SO2, CO, NO2, dan O3 pada hari berikutnya.
        """
        )

    elif selected == "Data":

        st.subheader("""Deskripsi Data""")
        st.write(
            """
        Data yang digunakan dalam aplikasi ini yaitu data ISPU DKI Jakarta periode 1 Desember 2022 sampai 30 November 2023. Data yang ditampilkan adalah data ispu yang diperoleh per harinya. 
        """
        )

        st.subheader("""Sumber Dataset""")
        st.write(
            """
        Sumber data didapatkan dari website "Satu Data DKI Jakarta". Berikut merupakan link untuk mengakses sumber dataset.
        <a href="https://satudata.jakarta.go.id/search?q=data%20ispu&halaman=all&kategori=all&topik=all&organisasi=all&status=all&sort=desc&page_no=1&show_filter=true&lang=id">Klik disini</a>""",
            unsafe_allow_html=True,
        )

        st.subheader("""Dataset""")
        # Menggunakan file Excel dari GitHub
        df = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        st.dataframe(df, width=600)
        
        st.subheader("Penghapusan kolom")
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        data.tanggal = pd.to_datetime(data.tanggal)
        
        # Menampilkan dataframe setelah penghapusan kolom
        st.dataframe(data, width=600)
        
    elif selected == "Missing Value":
        # MEAN IMPUTATION
        st.subheader("""Mean Imputation""")
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Menampilkan jumlah missing value per kolom
        missing_values = data.isnull().sum()
        st.write("Jumlah Missing Value per Kolom:")
        st.dataframe(missing_values[missing_values > 0].reset_index(name='missing_values'))
        
        # Mengidentifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Imputasi mean untuk kolom numerik
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Menampilkan data yang telah diproses
        st.dataframe(data, width=600)

        # PLOTING DATA
        st.subheader("""Ploting Data""")
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Mengidentifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Imputasi mean untuk kolom numerik
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom yang disebutkan ke tipe data integer
        data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']] = data[['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']].astype(int)
        
        # Resample data harian dan menghitung rata-rata
        data_resample = data.set_index('tanggal').resample('D').mean().reset_index()
        
        # Menentukan ukuran figure untuk subplot
        plt.figure(figsize=(12, 18))  # Ukuran figure untuk 6 subplot
        
        # Plot PM10
        plt.subplot(6, 1, 1)  # 6 baris, 1 kolom, subplot ke-1
        plt.plot(data_resample['tanggal'], data_resample['pm_sepuluh'], color='red')
        plt.title('Konsentrasi PM10')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot PM2.5
        plt.subplot(6, 1, 2)  # subplot ke-2
        plt.plot(data_resample['tanggal'], data_resample['pm_duakomalima'], color='yellow')
        plt.title('Konsentrasi PM2.5')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot Karbon Monoksida
        plt.subplot(6, 1, 3)  # subplot ke-3
        plt.plot(data_resample['tanggal'], data_resample['karbon_monoksida'], color='green')
        plt.title('Konsentrasi Karbon Monoksida')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot Ozon
        plt.subplot(6, 1, 4)  # subplot ke-4
        plt.plot(data_resample['tanggal'], data_resample['ozon'], color='magenta')
        plt.title('Konsentrasi Ozon')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot Nitrogen Dioksida
        plt.subplot(6, 1, 5)  # subplot ke-5
        plt.plot(data_resample['tanggal'], data_resample['nitrogen_dioksida'], color='black')
        plt.title('Konsentrasi Nitrogen Dioksida')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Plot Sulfur Dioksida
        plt.subplot(6, 1, 6)  # subplot ke-6
        plt.plot(data_resample['tanggal'], data_resample['sulfur_dioksida'], color='blue')
        plt.title('Konsentrasi Sulfur Dioksida')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi (µg/m³)')
        plt.grid()
        
        # Menyesuaikan layout untuk menghindari tumpang tindih
        plt.tight_layout()
        
        # Menampilkan plot
        plt.show()
        # Setelah plotting
        st.pyplot()

        # Standardisasi DATA
        # Normalisasi Data
        st.subheader("Normalisasi Data")
        scaler = MinMaxScaler()
        
        # Daftar kolom polutan
        pollutants = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
        
        # Melakukan normalisasi untuk setiap kolom polutan
        for col in pollutants:
            with st.expander(f"{col.upper()} - Normalisasi Data"):
                # Reshape data menjadi 2D array
                values = data[col].values.reshape(-1, 1)
                
                # Fit dan transform data
                normalized_values = scaler.fit_transform(values)
                
                # Masukkan hasil normalisasi ke dalam DataFrame
                data[f'{col}_normalized'] = normalized_values
                
                # Tampilkan hasil normalisasi
                st.write(data[[col, f'{col}_normalized']])
        
        # Tampilkan semua kolom polutan dan kolom normalisasi mereka
        st.write(data[[f'{col}' for col in pollutants] + [f'{col}_normalized' for col in pollutants]])

    elif selected == "Modeling":
        # Membaca dataset dari file Excel
        data = pd.read_excel(
            "https://raw.githubusercontent.com/shintaputrii/skripsi/main/kualitasudara.xlsx"
        )
        
        # Menghapus kolom yang tidak diinginkan
        data = data.drop(['periode_data', 'stasiun', 'parameter_pencemar_kritis', 'max', 'kategori'], axis=1)
        
        # Mengganti nilai '-' dengan NaN
        data.replace(r'-+', np.nan, regex=True, inplace=True)
        
        # Mengidentifikasi kolom numerik
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        # Imputasi mean untuk kolom numerik
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Konversi kolom PM10 ke tipe data integer
        data['pm_sepuluh'] = data['pm_sepuluh'].astype(int)
        
        # Bagian MODELLING
        st.subheader("Modelling PM10")
        
        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
        
        # Fungsi Fuzzy KNN untuk PM10
        def fuzzy_knn_predict(data, k=3, test_size=0.3):
            # Normalisasi data PM10
            imports = data['pm_sepuluh'].values.reshape(-1, 1)
            data['pm_sepuluh_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data['pm_sepuluh_normalized'].values[:-1].reshape(-1, 1)
            y = data['pm_sepuluh_normalized'].values[1:]
        
            # Bagi data menjadi train dan test sesuai rasio yang diberikan
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
            # Menyimpan tanggal untuk test set
            dates_test = data['tanggal'].values[-len(y_test):]
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(X_test, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(len(X_test))
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(len(X_test)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi dan nilai aktual ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
            # Menghitung MAPE
            mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
            
            # Menampilkan hasil prediksi dan nilai aktual
            results = pd.DataFrame({'Tanggal': dates_test, 'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
            st.write(f'MAPE untuk pembagian data {int((1-test_size)*100)}% - {int(test_size*100)}%: {mape:.2f}%')
            st.write("Hasil Prediksi:")
            st.write(results)
        
            return mape
        
        # Menyimpan MAPE untuk setiap rasio
        mapes = []
        test_sizes = [0.3, 0.2, 0.1]  # 70%-30%, 80%-20%, 90%-10%
        
        for test_size in test_sizes:
            mape = fuzzy_knn_predict(data, k=3, test_size=test_size)
            mapes.append(mape)

        # Modeling PM2.5
        st.subheader("Modelling PM2.5")

        # Fungsi untuk normalisasi data
        def normalize_data(data):
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data)
            return normalized_data, scaler
        
        # Fungsi untuk menghitung keanggotaan fuzzy (inverse distance)
        def calculate_membership_inverse(distances, epsilon=1e-10):
            memberships = 1 / (distances + epsilon)
            return memberships
            
        # Fungsi Fuzzy KNN untuk PM2.5
        def fuzzy_knn_predict_pm25(data, k=3, test_size=0.3):
            # Normalisasi data PM2.5
            imports = data['pm_duakomalima'].values.reshape(-1, 1)
            data['pm_duakomalima_normalized'], scaler = normalize_data(imports)
        
            # Ekstrak fitur dan target
            X = data['pm_duakomalima_normalized'].values[:-1].reshape(-1, 1)
            y = data['pm_duakomalima_normalized'].values[1:]
        
            # Bagi data menjadi train dan test sesuai rasio yang diberikan
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
            # Menyimpan tanggal untuk test set
            dates_test = data['tanggal'].values[-len(y_test):]
        
            # Inisialisasi model KNN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
        
            # Mendapatkan tetangga dan jaraknya
            distances, indices = knn.kneighbors(X_test, n_neighbors=k, return_distance=True)
        
            # Inisialisasi array untuk menyimpan prediksi
            y_pred = np.zeros(len(X_test))
        
            # Loop untuk menghitung prediksi berdasarkan membership
            for i in range(len(X_test)):
                neighbor_distances = distances[i]
                neighbor_indices = indices[i]
                neighbor_targets = y_train[neighbor_indices]
                memberships = calculate_membership_inverse(neighbor_distances)
                y_pred[i] = np.sum(memberships * neighbor_targets) / np.sum(memberships)
        
            # Mengembalikan nilai prediksi dan nilai aktual ke skala awal
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
            # Menghitung MAPE
            mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
            
            # Menampilkan hasil prediksi dan nilai aktual
            results = pd.DataFrame({'Tanggal': dates_test, 'Actual': y_test_original.flatten(), 'Predicted': y_pred_original.flatten()})
            st.write(f'MAPE untuk pembagian data {int((1-test_size)*100)}% - {int(test_size*100)}%: {mape:.2f}%')
            st.write("Hasil Prediksi:")
            st.write(results)
        
            return mape
        
        # Menyimpan MAPE untuk setiap rasio PM2.5
        mapes_pm25 = []
        test_sizes = [0.3, 0.2, 0.1]  # 70%-30%, 80%-20%, 90%-10%
        
        for test_size in test_sizes:
            mape = fuzzy_knn_predict_pm25(data, k=3, test_size=test_size)
            mapes_pm25.append(mape)

    elif selected == "Implementation":
        df = pd.read_csv(
            "https://raw.githubusercontent.com/normalitariyn/dataset/main/data%20ispu%20dki%20jakarta.csv"
        )

        # mean imputation
        data = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy="mean")
        imputed_data = imputer.fit_transform(df[data])
        imputed_df_numeric = pd.DataFrame(imputed_data, columns=data, index=df.index)
        imputed_df = pd.concat([imputed_df_numeric, df.drop(columns=data)], axis=1)
        imports_pm10 = imputed_df["pm10"].values
        imports_so2 = imports = imputed_df["so2"].values
        imports_co = imports = imputed_df["co"].values
        imports_no2 = imports = imputed_df["no2"].values
        imports_o3 = imports = imputed_df["o3"].values

        # univariate transform
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

        # transform to a supervised learning problem
        kolom = 4

        # standard scaler
        scaler = StandardScaler()

        # create Tabs
        tab_titles = ["PM10", "SO2", "‍CO", "O3", "NO2"]
        tabs = st.tabs(tab_titles)

        # PM10
        with tabs[0]:
            with st.form("Implementasi_pm10"):

                # univariate PM10
                X_pm10, y_pm10 = split_sequence(imports_pm10, kolom)
                shapeX_pm10 = X_pm10.shape
                dfX_pm10 = pd.DataFrame(X_pm10)
                dfy_pm10 = pd.DataFrame(y_pm10, columns=["Xt"])
                df_pm10 = pd.concat((dfX_pm10, dfy_pm10), axis=1)

                # standardisasi pm10
                scalerX = StandardScaler()
                scalerY = StandardScaler()
                scaledX_pm10 = scalerX.fit_transform(dfX_pm10)
                scaledY_pm10 = scalerY.fit_transform(dfy_pm10)
                features_namesX_pm10 = dfX_pm10.columns.copy()
                features_namesy_pm10 = dfy_pm10.columns.copy()
                scaled_featuresX_pm10 = pd.DataFrame(
                    scaledX_pm10, columns=features_namesX_pm10
                )
                scaled_featuresY_pm10 = pd.DataFrame(
                    scaledY_pm10, columns=features_namesy_pm10
                )

                # pembagian dataset pm10
                training, test = train_test_split(
                    scaled_featuresX_pm10, test_size=0.1, random_state=0, shuffle=False
                )
                training_label, test_label = train_test_split(
                    scaled_featuresY_pm10, test_size=0.1, random_state=0, shuffle=False
                )

                # Mengubah training_label pm10 ke bentuk array
                training_label = np.array(training_label).reshape(-1, 1)

                # Membuat model SVR
                regresor = SVR(kernel="rbf", C=10, gamma=0.01, epsilon=0.010076808)
                regresor.fit(training, training_label.ravel())

                st.subheader("Implementasi Peramalan PM10")
                v1 = st.number_input(
                    "Masukkan kadar konsentrasi PM10 pada 4 hari sebelumnya"
                )
                v2 = st.number_input(
                    "Masukkan kadar konsentrasi PM10 pada 3 hari sebelumnya"
                )
                v3 = st.number_input(
                    "Masukkan kadar konsentrasi PM10 pada 2 hari sebelumnya"
                )
                v4 = st.number_input(
                    "Masukkan kadar konsentrasi PM10 pada 1 hari sebelumnya"
                )

                periode_options = {
                    "3 Hari": 3,
                    "7 Hari": 7,
                }

                periode = st.selectbox(
                    "Pilih Periode Peramalan", list(periode_options.keys())
                )

                # submit inputan
                prediksi = st.form_submit_button("Submit")
                if prediksi:
                    inputs = np.array([v1, v2, v3, v4]).reshape(1, -1)

                    # normalisasi data input
                    scaler_input = StandardScaler().fit(dfX_pm10)
                    normalized_input = scaler_input.transform(inputs)

                    days_to_predict = periode_options[periode]
                    predictions = []
                    current_input = normalized_input

                    for day in range(days_to_predict):
                        # Predict the next day
                        next_pred = regresor.predict(current_input.reshape(1, -1))
                        predictions.append(next_pred[0])

                        # Prepare input for the next day
                        current_input = np.append(
                            current_input[:, 1:], next_pred
                        ).reshape(1, -1)

                    # Denormalize predictions
                    denormalized_predictions = scalerY.inverse_transform(
                        np.array(predictions).reshape(-1, 1)
                    ).flatten()

                    # Display predictions
                    st.header(f"Hasil Peramalan PM10 untuk {periode}")
                    if periode in ["7 Hari", "3 Hari"]:
                        for i, prediction in enumerate(
                            denormalized_predictions, start=1
                        ):
                            st.write(f"Hari ke-{i}: {prediction:.1f}")

        # SO2
        with tabs[1]:

            with st.form("Implementasi_so2"):

                # univariate SO2
                X_so2, y_so2 = split_sequence(imports_so2, kolom)
                shapeX_so2 = X_so2.shape
                dfX_so2 = pd.DataFrame(X_so2)
                dfy_so2 = pd.DataFrame(y_so2, columns=["Xt"])
                df_so2 = pd.concat((dfX_so2, dfy_so2), axis=1)

                # standardisasi SO2
                scalerX = StandardScaler()
                scalerY = StandardScaler()
                scaledX_so2 = scalerX.fit_transform(dfX_so2)
                scaledY_so2 = scalerY.fit_transform(dfy_so2)
                features_namesX_so2 = dfX_so2.columns.copy()
                features_namesy_so2 = dfy_so2.columns.copy()
                scaled_featuresX_so2 = pd.DataFrame(
                    scaledX_so2, columns=features_namesX_so2
                )
                scaled_featuresY_so2 = pd.DataFrame(
                    scaledY_so2, columns=features_namesy_so2
                )

                # pembagian dataset so2
                training, test = train_test_split(
                    scaled_featuresX_so2, test_size=0.1, random_state=0, shuffle=False
                )
                training_label, test_label = train_test_split(
                    scaled_featuresY_so2, test_size=0.1, random_state=0, shuffle=False
                )

                # Mengubah training_label so2 ke bentuk array
                training_label = np.array(training_label).reshape(-1, 1)

                # Membuat model SVR
                regresor = SVR(
                    kernel="rbf", gamma=0.999504820023307, C=0.01, epsilon=0.1
                )
                regresor.fit(training, training_label.ravel())

                st.subheader("Implementasi Peramalan SO2")
                v1 = st.number_input(
                    "Masukkan kadar konsentrasi SO2 pada 4 hari sebelumnya"
                )
                v2 = st.number_input(
                    "Masukkan kadar konsentrasi SO2 pada 3 hari sebelumnya"
                )
                v3 = st.number_input(
                    "Masukkan kadar konsentrasi SO2 pada 2 hari sebelumnya"
                )
                v4 = st.number_input(
                    "Masukkan kadar konsentrasi SO2 pada 1 hari sebelumnya"
                )

                periode_options = {
                    "3 Hari": 3,
                    "7 Hari": 7,
                }

                periode = st.selectbox(
                    "Pilih Periode Peramalan", list(periode_options.keys())
                )

                # submit inputan
                prediksi = st.form_submit_button("Submit")
                if prediksi:
                    inputs = np.array([v1, v2, v3, v4]).reshape(1, -1)

                    # normalisasi data input
                    scaler_input = StandardScaler().fit(dfX_so2)
                    normalized_input = scaler_input.transform(inputs)

                    days_to_predict = periode_options[periode]
                    predictions = []
                    current_input = normalized_input

                    for day in range(days_to_predict):
                        # Predict the next day
                        next_pred = regresor.predict(current_input.reshape(1, -1))
                        predictions.append(next_pred[0])

                        # Prepare input for the next day
                        current_input = np.append(
                            current_input[:, 1:], next_pred
                        ).reshape(1, -1)

                    # Denormalize predictions
                    denormalized_predictions = scalerY.inverse_transform(
                        np.array(predictions).reshape(-1, 1)
                    ).flatten()

                    # Display predictions
                    st.header(f"Hasil Peramalan SO2 untuk {periode}")
                    if periode in ["7 Hari", "3 Hari"]:
                        for i, prediction in enumerate(
                            denormalized_predictions, start=1
                        ):
                            st.write(f"Hari ke-{i}: {prediction:.1f}")

        # CO
        with tabs[2]:
            with st.form("Implementasi_co"):

                # univariate CO
                X_co, y_co = split_sequence(imports_co, kolom)
                shapeX_co = X_co.shape
                dfX_co = pd.DataFrame(X_co)
                dfy_co = pd.DataFrame(y_co, columns=["Xt"])
                df_co = pd.concat((dfX_co, dfy_co), axis=1)

                # standardisasi CO
                scalerX = StandardScaler()
                scalerY = StandardScaler()
                scaledX_co = scalerX.fit_transform(dfX_co)
                scaledY_co = scalerY.fit_transform(dfy_co)
                features_namesX_co = dfX_co.columns.copy()
                features_namesy_co = dfy_co.columns.copy()
                scaled_featuresX_co = pd.DataFrame(
                    scaledX_co, columns=features_namesX_co
                )
                scaled_featuresY_co = pd.DataFrame(
                    scaledY_co, columns=features_namesy_co
                )

                # pembagian dataset CO
                training, test = train_test_split(
                    scaled_featuresX_co, test_size=0.1, random_state=0, shuffle=False
                )
                training_label, test_label = train_test_split(
                    scaled_featuresY_co, test_size=0.1, random_state=0, shuffle=False
                )

                # Mengubah training_label CO ke bentuk array
                training_label = np.array(training_label).reshape(-1, 1)

                # Membuat model SVR
                regresor = SVR(
                    kernel="rbf", gamma=100, C=0.1, epsilon=0.0184040563866588
                )
                regresor.fit(training, training_label.ravel())

                st.subheader("Implementasi Peramalan CO")
                v1 = st.number_input(
                    "Masukkan kadar konsentrasi CO pada 4 hari sebelumnya"
                )
                v2 = st.number_input(
                    "Masukkan kadar konsentrasi CO pada 3 hari sebelumnya"
                )
                v3 = st.number_input(
                    "Masukkan kadar konsentrasi CO pada 2 hari sebelumnya"
                )
                v4 = st.number_input(
                    "Masukkan kadar konsentrasi cO pada 1 hari sebelumnya"
                )

                periode_options = {
                    "3 Hari": 3,
                    "7 Hari": 7,
                }

                periode = st.selectbox(
                    "Pilih Periode Peramalan", list(periode_options.keys())
                )

                # submit inputan
                prediksi = st.form_submit_button("Submit")
                if prediksi:
                    inputs = np.array([v1, v2, v3, v4]).reshape(1, -1)

                    # normalisasi data input
                    scaler_input = StandardScaler().fit(dfX_co)
                    normalized_input = scaler_input.transform(inputs)

                    days_to_predict = periode_options[periode]
                    predictions = []
                    current_input = normalized_input

                    for day in range(days_to_predict):
                        # Predict the next day
                        next_pred = regresor.predict(current_input.reshape(1, -1))
                        predictions.append(next_pred[0])

                        # Prepare input for the next day
                        current_input = np.append(
                            current_input[:, 1:], next_pred
                        ).reshape(1, -1)

                    # Denormalize predictions
                    denormalized_predictions = scalerY.inverse_transform(
                        np.array(predictions).reshape(-1, 1)
                    ).flatten()

                    # Display predictions
                    st.header(f"Hasil Peramalan CO untuk {periode}")
                    if periode in ["7 Hari", "3 Hari"]:
                        for i, prediction in enumerate(
                            denormalized_predictions, start=1
                        ):
                            st.write(f"Hari ke-{i}: {prediction:.1f}")

        # O3
        with tabs[3]:

            with st.form("Implementasi_o3"):

                # univariate O3
                X_o3, y_o3 = split_sequence(imports_o3, kolom)
                shapeX_o3 = X_o3.shape
                dfX_o3 = pd.DataFrame(X_o3)
                dfy_o3 = pd.DataFrame(y_o3, columns=["Xt"])
                df_o3 = pd.concat((dfX_o3, dfy_o3), axis=1)

                # standardisasi O3
                scalerX = StandardScaler()
                scalerY = StandardScaler()
                scaledX_o3 = scalerX.fit_transform(dfX_o3)
                scaledY_o3 = scalerY.fit_transform(dfy_o3)
                features_namesX_o3 = dfX_o3.columns.copy()
                features_namesy_o3 = dfy_o3.columns.copy()
                scaled_featuresX_o3 = pd.DataFrame(
                    scaledX_o3, columns=features_namesX_o3
                )
                scaled_featuresY_o3 = pd.DataFrame(
                    scaledY_o3, columns=features_namesy_o3
                )

                # pembagian dataset o3
                training, test = train_test_split(
                    scaled_featuresX_o3, test_size=0.1, random_state=0, shuffle=False
                )
                training_label, test_label = train_test_split(
                    scaled_featuresY_o3, test_size=0.1, random_state=0, shuffle=False
                )

                # Mengubah training_label o3 ke bentuk array
                training_label = np.array(training_label).reshape(-1, 1)

                # Membuat model SVR
                regresor = SVR(kernel="linear", C=10, epsilon=0.0105339836281039)
                regresor.fit(training, training_label.ravel())

                st.subheader("Implementasi Peramalan O3")
                v1 = st.number_input(
                    "Masukkan kadar konsentrasi O3 pada 4 hari sebelumnya"
                )
                v2 = st.number_input(
                    "Masukkan kadar konsentrasi O3 pada 3 hari sebelumnya"
                )
                v3 = st.number_input(
                    "Masukkan kadar konsentrasi O3 pada 2 hari sebelumnya"
                )
                v4 = st.number_input(
                    "Masukkan kadar konsentrasi O3 pada 1 hari sebelumnya"
                )

                periode_options = {
                    "3 Hari": 3,
                    "7 Hari": 7,
                }

                periode = st.selectbox(
                    "Pilih Periode Peramalan", list(periode_options.keys())
                )

                # submit inputan
                prediksi = st.form_submit_button("Submit")
                if prediksi:
                    inputs = np.array([v1, v2, v3, v4]).reshape(1, -1)

                    # normalisasi data input
                    scaler_input = StandardScaler().fit(dfX_o3)
                    normalized_input = scaler_input.transform(inputs)

                    days_to_predict = periode_options[periode]
                    predictions = []
                    current_input = normalized_input

                    for day in range(days_to_predict):
                        # Predict the next day
                        next_pred = regresor.predict(current_input.reshape(1, -1))
                        predictions.append(next_pred[0])

                        # Prepare input for the next day
                        current_input = np.append(
                            current_input[:, 1:], next_pred
                        ).reshape(1, -1)

                    # Denormalize predictions
                    denormalized_predictions = scalerY.inverse_transform(
                        np.array(predictions).reshape(-1, 1)
                    ).flatten()

                    # Display predictions
                    st.header(f"Hasil Peramalan O3 untuk {periode}")
                    if periode in ["7 Hari", "3 Hari"]:
                        for i, prediction in enumerate(
                            denormalized_predictions, start=1
                        ):
                            st.write(f"Hari ke-{i}: {prediction:.1f}")

        # NO2
        with tabs[4]:

            with st.form("Implementasi_no2"):

                # univariate NO2
                X_no2, y_no2 = split_sequence(imports_no2, kolom)
                shapeX_no2 = X_no2.shape
                dfX_no2 = pd.DataFrame(X_no2)
                dfy_no2 = pd.DataFrame(y_no2, columns=["Xt"])
                df_no2 = pd.concat((dfX_no2, dfy_no2), axis=1)

                # standardisasi NO2
                scalerX = StandardScaler()
                scalerY = StandardScaler()
                scaledX_no2 = scalerX.fit_transform(dfX_no2)
                scaledY_no2 = scalerY.fit_transform(dfy_no2)
                features_namesX_no2 = dfX_no2.columns.copy()
                features_namesy_no2 = dfy_no2.columns.copy()
                scaled_featuresX_no2 = pd.DataFrame(
                    scaledX_no2, columns=features_namesX_no2
                )
                scaled_featuresY_no2 = pd.DataFrame(
                    scaledY_no2, columns=features_namesy_no2
                )

                # pembagian dataset NO2
                training, test = train_test_split(
                    scaled_featuresX_no2, test_size=0.1, random_state=0, shuffle=False
                )
                training_label, test_label = train_test_split(
                    scaled_featuresY_no2, test_size=0.1, random_state=0, shuffle=False
                )

                # Mengubah training_label NO2 ke bentuk array
                training_label = np.array(training_label).reshape(-1, 1)

                # Membuat model SVR
                regresor = SVR(
                    kernel="linear", C=5.06915442747395, epsilon=0.0946570714100452
                )
                regresor.fit(training, training_label.ravel())

                st.subheader("Implementasi Peramalan NO2")
                v1 = st.number_input(
                    "Masukkan kadar konsentrasi NO2 pada 4 hari sebelumnya"
                )
                v2 = st.number_input(
                    "Masukkan kadar konsentrasi NO2 pada 3 hari sebelumnya"
                )
                v3 = st.number_input(
                    "Masukkan kadar konsentrasi NO2 pada 2 hari sebelumnya"
                )
                v4 = st.number_input(
                    "Masukkan kadar konsentrasi NO2 pada 1 hari sebelumnya"
                )

                periode_options = {
                    "3 Hari": 3,
                    "7 Hari": 7,
                }

                periode = st.selectbox(
                    "Pilih Periode Peramalan", list(periode_options.keys())
                )

                # submit inputan
                prediksi = st.form_submit_button("Submit")
                if prediksi:
                    inputs = np.array([v1, v2, v3, v4]).reshape(1, -1)

                    # normalisasi data input
                    scaler_input = StandardScaler().fit(dfX_no2)
                    normalized_input = scaler_input.transform(inputs)

                    days_to_predict = periode_options[periode]
                    predictions = []
                    current_input = normalized_input

                    for day in range(days_to_predict):
                        # Predict the next day
                        next_pred = regresor.predict(current_input.reshape(1, -1))
                        predictions.append(next_pred[0])

                        # Prepare input for the next day
                        current_input = np.append(
                            current_input[:, 1:], next_pred
                        ).reshape(1, -1)

                    # Denormalize predictions
                    denormalized_predictions = scalerY.inverse_transform(
                        np.array(predictions).reshape(-1, 1)
                    ).flatten()

                    # Display predictions
                    st.header(f"Hasil Peramalan NO2 untuk {periode}")
                    if periode in ["7 Hari", "3 Hari"]:
                        for i, prediction in enumerate(
                            denormalized_predictions, start=1
                        ):
                            st.write(f"Hari ke-{i}: {prediction:.1f}")

    elif selected == "About Me":
        st.write("Normalita Eka Ariyanti \n (200411100084) \n Teknik Informatika")
