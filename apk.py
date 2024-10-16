import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numpy import array
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Prediksi Kualitas Udara DKI Jakarta",
    page_icon="https://www.google.com/url?sa=i&url=https%3A%2F%2Fnews.detik.com%2Fberita%2Fd-6184561%2Fkualitas-udara-jakarta-hari-ini-terburuk-se-ri-versi-iqair&psig=AOvVaw0QJXZklBKL9IK98ihm3QAv&ust=1729164098523000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCJDFwcnkkokDFQAAAAAdAAAAABAE",
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
                """<h2 style = "text-align: center;"><img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fnews.detik.com%2Fberita%2Fd-6184561%2Fkualitas-udara-jakarta-hari-ini-terburuk-se-ri-versi-iqair&psig=AOvVaw0QJXZklBKL9IK98ihm3QAv&ust=1729164098523000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCJDFwcnkkokDFQAAAAAdAAAAABAE" width="130" height="130"><br></h2>""",
                unsafe_allow_html=True,
            ),
            [
                "Home",
                "Data",
                "Preprocessing",
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
        <img src="https://images.forestdigest.com/upload/2023/20230820121017.jpg" width="500" height="300">
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
        Data yang digunakan dalam aplikasi ini yaitu data ISPU DKI Jakarta periode 1 Januari 2015 sampai 31 Desember 2021. Data yang ditampilkan adalah data ispu yang diperoleh per harinya. 
        """
        )

        st.subheader("""Sumber Dataset""")
        st.write(
            """
        Sumber data didapatkan dari website "Satu Data DKI Jakarta". Berikut merupakan link untuk mengakses sumber dataset.
        <a href="https://satudata.jakarta.go.id/search?q=data%20ispu&halaman=all&kategori=all&topik=all&organisasi=all&status=all&sort=desc&page_no=1&show_filter=true&lang=id">Klik disini</a>""",
            unsafe_allow_html=True,
        )

        st.subheader("""Dataset ISPU DKI Jakarta""")
        df = pd.read_csv(
            "https://raw.githubusercontent.com/normalitariyn/dataset/main/data%20ispu%20dki%20jakarta.csv"
        )
        st.dataframe(df, width=600)

        st.subheader("""Statistik Missing Value""")
        df["tanggal"] = pd.to_datetime(df["tanggal"])
        df_clean = df.dropna(subset=["tanggal"])
        missing_value = df_clean.drop(columns=["tanggal"]).isnull().sum()
        missing_value = pd.DataFrame(missing_value, columns=["Jumlah Missing Value"])
        st.dataframe(missing_value, width=600)

        st.markdown("---")
        st.subheader("""Plotting Dataset""")
        df_resample = df_clean.set_index("tanggal").resample("D").mean().reset_index()

        # Function to plot each parameter
        def plot_parameter(data, parameter, color):
            fig, ax = plt.subplots()
            data.plot(x="tanggal", y=parameter, kind="line", color=color, ax=ax)
            ax.set_xlabel("Tanggal")
            ax.set_ylabel(parameter.upper())
            ax.set_title(f"{parameter.upper()} dari Waktu ke Waktu")
            return fig

        # Tampilkan plot masing-masing parameter
        st.subheader("Plot PM10")
        st.pyplot(plot_parameter(df_resample, "pm10", "red"))

        st.markdown("---")
        st.subheader("Plot SO2")
        st.pyplot(plot_parameter(df_resample, "so2", "green"))

        st.markdown("---")
        st.subheader("Plot CO")
        st.pyplot(plot_parameter(df_resample, "co", "magenta"))

        st.markdown("---")
        st.subheader("Plot O3")
        st.pyplot(plot_parameter(df_resample, "o3", "black"))

        st.markdown("---")
        st.subheader("Plot NO2")
        st.pyplot(plot_parameter(df_resample, "no2", "blue"))

    elif selected == "Preprocessing":
        # MEAN IMPUTATION
        st.subheader("""Mean Imputation""")
        df = pd.read_csv(
            "https://raw.githubusercontent.com/normalitariyn/dataset/main/data%20ispu%20dki%20jakarta.csv"
        )
        data = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy="mean")
        imputed_data = imputer.fit_transform(df[data])
        imputed_df_numeric = pd.DataFrame(imputed_data, columns=data, index=df.index)
        imputed_df = pd.concat([imputed_df_numeric, df.drop(columns=data)], axis=1)
        st.dataframe(imputed_df, width=600)

        # UNIVARIATE TRANSFORM
        st.subheader("""Univariate Transform""")
        imports_pm10 = imputed_df["pm10"].values
        imports_so2 = imports = imputed_df["so2"].values
        imports_co = imports = imputed_df["co"].values
        imports_no2 = imports = imputed_df["no2"].values
        imports_o3 = imports = imputed_df["o3"].values

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

        # define univariate time series
        series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # transform to a supervised learning problem
        kolom = 4

        with st.expander("PM10 - Univariate Transform"):
            X_pm10, y_pm10 = split_sequence(imports_pm10, kolom)
            print(X_pm10.shape, y_pm10.shape)
            shapeX_pm10 = X_pm10.shape
            dfX_pm10 = pd.DataFrame(X_pm10, columns=["t-4", "t-3", "t-2", "t-1"])
            dfy_pm10 = pd.DataFrame(y_pm10, columns=["Xt"])
            df_pm10 = pd.concat((dfX_pm10, dfy_pm10), axis=1)
            st.dataframe(df_pm10, width=600)

        with st.expander("SO2 - Univariate Transform"):
            X_so2, y_so2 = split_sequence(imports_so2, kolom)
            print(X_so2.shape, y_so2.shape)
            shapeX_so2 = X_so2.shape
            dfX_so2 = pd.DataFrame(X_so2, columns=["t-4", "t-3", "t-2", "t-1"])
            dfy_so2 = pd.DataFrame(y_so2, columns=["Xt"])
            df_so2 = pd.concat((dfX_so2, dfy_so2), axis=1)
            st.dataframe(df_so2, width=600)

        with st.expander("CO - Univariate Transform"):
            X_co, y_co = split_sequence(imports_co, kolom)
            print(X_co.shape, y_co.shape)
            shapeX_co = X_co.shape
            dfX_co = pd.DataFrame(X_co, columns=["t-4", "t-3", "t-2", "t-1"])
            dfy_co = pd.DataFrame(y_co, columns=["Xt"])
            df_co = pd.concat((dfX_co, dfy_co), axis=1)
            st.dataframe(df_co, width=600)

        with st.expander("NO2 - Univariate Transform"):
            X_no2, y_no2 = split_sequence(imports_no2, kolom)
            print(X_no2.shape, y_no2.shape)
            shapeX_no2 = X_no2.shape
            dfX_no2 = pd.DataFrame(X_no2, columns=["t-4", "t-3", "t-2", "t-1"])
            dfy_no2 = pd.DataFrame(y_no2, columns=["Xt"])
            df_no2 = pd.concat((dfX_no2, dfy_no2), axis=1)
            st.dataframe(df_no2, width=600)

        with st.expander("O3 - Univariate Transform"):
            X_o3, y_o3 = split_sequence(imports_o3, kolom)
            print(X_o3.shape, y_o3.shape)
            shapeX_o3 = X_o3.shape
            dfX_o3 = pd.DataFrame(X_o3, columns=["t-4", "t-3", "t-2", "t-1"])
            dfy_o3 = pd.DataFrame(y_o3, columns=["Xt"])
            df_o3 = pd.concat((dfX_o3, dfy_o3), axis=1)
            st.dataframe(df_o3, width=600)

        # Standardisasi DATA
        st.subheader("""Standardisasi Data""")
        scaler = StandardScaler()

        with st.expander("PM10 - Normalisasi Data"):
            scaledX_pm10 = scaler.fit_transform(dfX_pm10)
            scaledY_pm10 = scaler.fit_transform(dfy_pm10)
            features_namesX_pm10 = dfX_pm10.columns.copy()
            features_namesy_pm10 = dfy_pm10.columns.copy()
            # features_names.remove('label')
            scaled_featuresX_pm10 = pd.DataFrame(
                scaledX_pm10, columns=features_namesX_pm10
            )
            scaled_featuresY_pm10 = pd.DataFrame(
                scaledY_pm10, columns=features_namesy_pm10
            )
            normalized_pm10 = pd.concat(
                (scaled_featuresX_pm10, scaled_featuresY_pm10), axis=1
            )
            st.dataframe(normalized_pm10, width=600)

        with st.expander("SO2 - Normalisasi Data"):
            scaledX_so2 = scaler.fit_transform(dfX_so2)
            scaledY_so2 = scaler.fit_transform(dfy_so2)
            features_namesX_so2 = dfX_so2.columns.copy()
            features_namesy_so2 = dfy_so2.columns.copy()
            # features_names.remove('label')
            scaled_featuresX_so2 = pd.DataFrame(
                scaledX_so2, columns=features_namesX_so2
            )
            scaled_featuresY_so2 = pd.DataFrame(
                scaledY_so2, columns=features_namesy_so2
            )
            normalized_so2 = pd.concat(
                (scaled_featuresX_so2, scaled_featuresY_so2), axis=1
            )
            st.dataframe(normalized_so2, width=600)

        with st.expander("CO - Normalisasi Data"):
            scaledX_co = scaler.fit_transform(dfX_co)
            scaledY_co = scaler.fit_transform(dfy_co)
            features_namesX_co = dfX_co.columns.copy()
            features_namesy_co = dfy_co.columns.copy()
            # features_names.remove('label')
            scaled_featuresX_co = pd.DataFrame(scaledX_co, columns=features_namesX_co)
            scaled_featuresY_co = pd.DataFrame(scaledY_co, columns=features_namesy_co)
            normalized_co = pd.concat(
                (scaled_featuresX_co, scaled_featuresY_co), axis=1
            )
            st.dataframe(normalized_co, width=600)

        with st.expander("NO2 - Normalisasi Data"):
            scaledX_no2 = scaler.fit_transform(dfX_no2)
            scaledY_no2 = scaler.fit_transform(dfy_no2)
            features_namesX_no2 = dfX_no2.columns.copy()
            features_namesy_no2 = dfy_no2.columns.copy()
            # features_names.remove('label')
            scaled_featuresX_no2 = pd.DataFrame(
                scaledX_no2, columns=features_namesX_no2
            )
            scaled_featuresY_no2 = pd.DataFrame(
                scaledY_no2, columns=features_namesy_no2
            )
            normalized_no2 = pd.concat(
                (scaled_featuresX_no2, scaled_featuresY_no2), axis=1
            )
            st.dataframe(normalized_no2, width=600)

        with st.expander("O3 - Normalisasi Data"):
            scaledX_o3 = scaler.fit_transform(dfX_o3)
            scaledY_o3 = scaler.fit_transform(dfy_o3)
            features_namesX_o3 = dfX_o3.columns.copy()
            features_namesy_o3 = dfy_o3.columns.copy()
            # features_names.remove('label')
            scaled_featuresX_o3 = pd.DataFrame(scaledX_o3, columns=features_namesX_o3)
            scaled_featuresY_o3 = pd.DataFrame(scaledY_o3, columns=features_namesy_o3)
            normalized_o3 = pd.concat(
                (scaled_featuresX_o3, scaled_featuresY_o3), axis=1
            )
            st.dataframe(normalized_o3, width=600)

    elif selected == "Modeling":
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

        # define univariate time series
        series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # transform to a supervised learning problem
        kolom = 4

        # standard scaler
        scaler = StandardScaler()

        # create Tabs
        tab_titles = ["PM10", "SO2", "‍CO", "NO2", "O3"]
        tabs = st.tabs(tab_titles)

        # PM10
        with tabs[0]:
            st.markdown(
                "<h1 style='text-align: center; '>Modelling PM10</h2>",
                unsafe_allow_html=True,
            )
            selected_pm10 = st.selectbox(
                "Pilih Modelling", ["Select Modelling PM10", "SVR PM10", "SVR-PSO PM10"]
            )

            # univariate pm10
            X_pm10, y_pm10 = split_sequence(imports_pm10, kolom)
            print(X_pm10.shape, y_pm10.shape)
            shapeX_pm10 = X_pm10.shape
            dfX_pm10 = pd.DataFrame(X_pm10)
            dfy_pm10 = pd.DataFrame(y_pm10, columns=["Xt"])
            df_pm10 = pd.concat((dfX_pm10, dfy_pm10), axis=1)

            # standardisasi pm10
            scaledX_pm10 = scaler.fit_transform(dfX_pm10)
            scaledY_pm10 = scaler.fit_transform(dfy_pm10)
            features_namesX_pm10 = dfX_pm10.columns.copy()
            features_namesy_pm10 = dfy_pm10.columns.copy()
            scaled_featuresX_pm10 = pd.DataFrame(
                scaledX_pm10, columns=features_namesX_pm10
            )
            scaled_featuresY_pm10 = pd.DataFrame(
                scaledY_pm10, columns=features_namesy_pm10
            )

            if selected_pm10 == "SVR PM10":

                # Input kernel dan test size PM10
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_pm10",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_pm10",
                    )

                # Pastikan nilai test_size dan kernel valid
                if test_size != "Select" and kernel != "Select":

                    # pembagian dataset PM10
                    test_size = float(test_size)  # Konversi test_size ke float
                    training, test = train_test_split(
                        scaled_featuresX_pm10,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_pm10,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    # Mengubah training_label pm10 ke bentuk array
                    training_label = np.array(training_label).reshape(-1, 1)

                    # Membuat model SVR
                    regresor = SVR(kernel=kernel, C=1, gamma="scale", epsilon=0.1)
                    regresor.fit(training, training_label.ravel())

                    # Prediksi data testing
                    pred_test = regresor.predict(test)
                    pred_test = pred_test.reshape(-1, 1)

                    # Denormalisasi data testing
                    denormalized_train_pm10 = pd.DataFrame(
                        scaler.inverse_transform(test_label),
                        columns=["Data Aktual PM10 (Testing)"],
                    )
                    denormalized_pred_pm10 = pd.DataFrame(
                        scaler.inverse_transform(pred_test),
                        columns=["Data Prediksi PM10 (Testing)"],
                    )
                    hasil_pm10 = pd.concat(
                        [denormalized_train_pm10, denormalized_pred_pm10], axis=1
                    )

                    # Hitung MAPE
                    MAPE = mean_absolute_percentage_error(
                        denormalized_train_pm10, denormalized_pred_pm10
                    )

                    # Menampilkan hasil prediksi dan data aktual test data
                    st.subheader("Prediksi Data Testing")
                    st.write(hasil_pm10)

                    # # Menampilkan plot antara data aktual dan data prediksi pada data test
                    # st.subheader("Plotting Data Aktual VS Data Prediksi - Data Testing")
                    # st.line_chart(hasil_pm10)

                    # plotting data
                    st.subheader("Plotting Data Aktual VS Data Prediksi - Data Testing")
                    fig = go.Figure()
                    # Menambahkan data aktual dengan warna biru tua
                    fig.add_trace(
                        go.Scatter(
                            x=hasil_pm10.index,
                            y=hasil_pm10["Data Aktual PM10 (Testing)"],
                            mode="lines",
                            name="Data Aktual",
                            line=dict(color="darkblue"),
                        )
                    )
                    # Menambahkan data prediksi dengan warna merah
                    fig.add_trace(
                        go.Scatter(
                            x=hasil_pm10.index,
                            y=hasil_pm10["Data Prediksi PM10 (Testing)"],
                            mode="lines",
                            name="Data Prediksi",
                            line=dict(color="red"),
                        )
                    )
                    # Mengatur layout figure untuk memperbesar ukuran grafik
                    fig.update_layout(
                        width=800, height=600  # Lebar figure  # Tinggi figure
                    )
                    st.plotly_chart(fig, use_container_width=False)

                    # Menampilkan matriks evaluasi
                    st.subheader("Metriks Evaluasi Data Testing")
                    st.info(f"MAPE :\n{MAPE*100}%")

                else:
                    st.warning("Please select both a test size and a kernel.")

            elif selected_pm10 == "SVR-PSO PM10":
                # Input kernel dan test size PM10
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_pm10",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_pm10",
                    )

                # Input pop size dan max iter PSO
                col1, col2 = st.columns(2)
                with col1:
                    popsize = st.selectbox(
                        "popsize",
                        ["Select", "5", "10", "20", "30"],
                        key="popsize_pm10",
                    )

                with col2:
                    max_iter = st.selectbox(
                        "max_iter",
                        ["Select", "10", "20", "50", "100"],
                        key="max_iter_pm10",
                    )

                # buat kolom
                if (
                    test_size != "Select"
                    and kernel != "Select"
                    and popsize != "Select"
                    and max_iter != "Select"
                ):

                    # konversi select data ke dalam bentuk numeric
                    test_size = float(test_size)
                    popsize = int(popsize)
                    max_iter = int(max_iter)

                    # pembagian data
                    training, test = train_test_split(
                        scaled_featuresX_pm10,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_pm10,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    st.markdown(
                        "<h2>Hasil Parameter PSO-SVR</h2>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"test size : {test_size}, kernel : {kernel}, popsize : {popsize}, max iter : {max_iter}"
                    )

                    # Define the bounds for each hyperparameter
                    if kernel == "linear":
                        lower_bound = [0.01, 0.01]  # C and epsilon only
                        upper_bound = [10, 0.1]  # C and epsilon only
                        n_dimensions = 2
                    else:
                        lower_bound = [0.01, 0.01, 0.01]
                        upper_bound = [10, 0.1, 0.1]
                        n_dimensions = 3

                    # Define the PSO algorithm
                    w = 0.5
                    c1 = 2
                    c2 = 2

                    def fitness_function(mape):
                        return 1 / (mape + 1)

                    # Define the objective function
                    # Set random seed for reproducibility
                    np.random.seed(42)

                    def objective_function(params):
                        if kernel == "linear":
                            c, epsilon = params
                            svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                        else:
                            c, gamma, epsilon = params
                            svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                        svr.fit(training, training_label)
                        pred_train = svr.predict(training)
                        pred_train1 = pred_train.reshape(-1, 1)

                        # Denormalisasi Data
                        denormalized_data_train = pd.DataFrame(
                            scaler.inverse_transform(
                                training_label.values.reshape(-1, 1)
                            ),
                            columns=["Testing Data"],
                        )
                        denormalized_data_preds = pd.DataFrame(
                            scaler.inverse_transform(pred_train1),
                            columns=["Predict Data"],
                        )
                        mape = mean_absolute_percentage_error(
                            denormalized_data_train, denormalized_data_preds
                        )
                        return mape

                    def pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    ):

                        particles = np.random.uniform(
                            low=lower_bound,
                            high=upper_bound,
                            size=(n_particles, n_dimensions),
                        )

                        personal_best_positions = particles.copy()
                        personal_best_scores = [
                            objective_function(p) for p in personal_best_positions
                        ]
                        global_best_position = particles[
                            np.argmin(personal_best_scores)
                        ]
                        global_best_error = min(personal_best_scores)
                        global_best_fitness = fitness_function(global_best_error)

                        velocities = np.zeros((n_particles, n_dimensions))

                        all_particles = []
                        all_mape = []
                        all_fitness = []
                        convergence_iter = None

                        for i in range(max_iter):

                            r1 = np.random.rand(n_particles, n_dimensions)
                            r2 = np.random.rand(n_particles, n_dimensions)
                            velocities = (
                                w * velocities
                                + c1 * r1 * (personal_best_positions - particles)
                                + c2 * r2 * (global_best_position - particles)
                            )

                            particles = particles + velocities

                            particles = np.clip(particles, lower_bound, upper_bound)

                            mape_particles = [objective_function(p) for p in particles]
                            fitness_particles = [
                                fitness_function(mape) for mape in mape_particles
                            ]
                            all_mape.append(mape_particles)
                            all_fitness.append(fitness_particles)

                            for j in range(n_particles):
                                error = mape_particles[j]
                                if error < personal_best_scores[j]:
                                    personal_best_positions[j] = particles[j]
                                    personal_best_scores[j] = error
                                    if error < global_best_error:
                                        global_best_position = particles[j]
                                        global_best_error = error
                                        global_best_fitness = fitness_function(
                                            global_best_error
                                        )

                            all_particles.append(particles.copy())

                            st.subheader(f"\nIteration {i + 1}:")
                            if kernel == "linear":
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Epsilon={global_best_position[1]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Epsilon={pbest[1]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )
                            else:
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Gamma={global_best_position[1]}, Epsilon={global_best_position[2]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Gamma={pbest[1]}, Epsilon={pbest[2]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )

                            if (
                                len(set(mape_particles)) == 1
                                and convergence_iter is None
                            ):
                                convergence_iter = i + 1

                        # Print the best particle found
                        st.markdown(
                            "<h3>Partikel Terbaik</h3>",
                            unsafe_allow_html=True,
                        )
                        if kernel == "linear":
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Epsilon: {global_best_position[1]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Gamma: {global_best_position[1]}</p>
                                    <p>Epsilon: {global_best_position[2]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        if convergence_iter is not None:
                            st.success(
                                f"\nConvergence reached at iteration {convergence_iter}"
                            )

                        return global_best_position

                    # SVR-PSO
                    # Define the PSO algorithm parameters
                    n_particles = popsize
                    max_iter = max_iter

                    # Call the PSO function with the specified hyperparameters
                    hyperparameters = pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    )

                    # Train the SVR model with the best hyperparameters and plot the results
                    if kernel == "linear":
                        c, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                    else:
                        c, gamma, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                    svr.fit(training, training_label)
                    pred_test = svr.predict(test)
                    pred_test1 = pred_test.reshape(-1, 1)

                    # Denormalisasi Data

                    test_label_array = (
                        test_label.values
                    )  # Convert DataFrame to NumPy array
                    reshaped_test_label = test_label_array.reshape(-1, 1)
                    denormalized_data_test = scaler.inverse_transform(
                        reshaped_test_label
                    )
                    denormalized_data_test = pd.DataFrame(
                        denormalized_data_test, columns=["Testing Data"]
                    )
                    denormalized_preds_test = pd.DataFrame(
                        scaler.inverse_transform(pred_test1), columns=["Predict Data"]
                    )
                    hasil_pm10_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )

                    # Pengujian dengan MAPE
                    MAPE = (
                        mean_absolute_percentage_error(
                            denormalized_data_test, denormalized_preds_test
                        )
                        * 100
                    )

                    # Menampilkan hasil prediksi dan data aktual training data
                    st.subheader("Hasil Prediksi Data Uji")
                    st.write(hasil_pm10_2)

                    # Plot the actual data vs predicted data
                    st.subheader("Plotting Hasil Prediksi dan Data Uji PM10")
                    chart_data_pm10_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )
                    st.line_chart(chart_data_pm10_2)

                    # Menampilkan Hasil Pengujian
                    st.subheader("Hasil Pengujian Data Uji")
                    st.write("MAPE")
                    st.info(f"Hasil MAPE : {MAPE}%")

        # SO2
        with tabs[1]:
            st.markdown(
                "<h1 style='text-align: center; '>Modelling SO2</h2>",
                unsafe_allow_html=True,
            )
            selected_so2 = st.selectbox(
                "Pilih Modelling", ["Select Modelling SO2", "SVR SO2", "SVR-PSO SO2"]
            )

            # univariate so2
            X_so2, y_so2 = split_sequence(imports_so2, kolom)
            print(X_so2.shape, y_so2.shape)
            shapeX_so2 = X_so2.shape
            dfX_so2 = pd.DataFrame(X_so2)
            dfy_so2 = pd.DataFrame(y_so2, columns=["Xt"])
            df_so2 = pd.concat((dfX_so2, dfy_so2), axis=1)

            # standardisasi so2
            scaledX_so2 = scaler.fit_transform(dfX_so2)
            scaledY_so2 = scaler.fit_transform(dfy_so2)
            features_namesX_so2 = dfX_so2.columns.copy()
            features_namesy_so2 = dfy_so2.columns.copy()
            scaled_featuresX_so2 = pd.DataFrame(
                scaledX_so2, columns=features_namesX_so2
            )
            scaled_featuresY_so2 = pd.DataFrame(
                scaledY_so2, columns=features_namesy_so2
            )

            if selected_so2 == "SVR SO2":

                # Input kernel dan test size SO2
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_so2",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_so2",
                    )

                # Pastikan nilai test_size dan kernel valid
                if test_size != "Select" and kernel != "Select":

                    # pembagian dataset SO2
                    test_size = float(test_size)  # Konversi test_size ke float
                    training, test = train_test_split(
                        scaled_featuresX_so2,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_so2,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    # Mengubah training_label so2 ke bentuk array
                    training_label = np.array(training_label).reshape(-1, 1)

                    # Membuat model SVR
                    regresor = SVR(kernel=kernel, C=1, gamma="scale", epsilon=0.1)
                    regresor.fit(training, training_label.ravel())

                    # Prediksi data testing
                    pred_test = regresor.predict(test)
                    pred_test = pred_test.reshape(-1, 1)

                    # Denormalisasi data testing
                    denormalized_test_so2 = pd.DataFrame(
                        scaler.inverse_transform(test_label),
                        columns=["Data Aktual SO2 (Testing)"],
                    )
                    denormalized_pred_so2 = pd.DataFrame(
                        scaler.inverse_transform(pred_test),
                        columns=["Data Prediksi SO2 (Testing)"],
                    )
                    hasil_so2 = pd.concat(
                        [denormalized_test_so2, denormalized_pred_so2], axis=1
                    )

                    # Hitung MAPE
                    MAPE = mean_absolute_percentage_error(
                        denormalized_test_so2, denormalized_pred_so2
                    )

                    # Menampilkan hasil prediksi dan data aktual training data
                    st.subheader("Prediksi Data Testing")
                    st.write(hasil_so2)

                    # Menampilkan plot antara data aktual dan data prediksi pada data training
                    st.subheader("Plotting Data Aktual VS Data Prediksi - Data Testing")
                    st.line_chart(hasil_so2)

                    # Menampilkan matriks evaluasi
                    st.subheader("Metriks Evaluasi Data Testing")
                    st.info(f"MAPE :\n{MAPE*100}%")

                else:
                    st.warning("Please select both a test size and a kernel.")

            elif selected_so2 == "SVR-PSO SO2":
                # Input kernel dan test size SO2
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_so2",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_so2",
                    )

                # Input pop size dan max iter PSO
                col1, col2 = st.columns(2)
                with col1:
                    popsize = st.selectbox(
                        "popsize",
                        ["Select", "5", "10", "20", "30"],
                        key="popsize_so2",
                    )

                with col2:
                    max_iter = st.selectbox(
                        "max_iter",
                        ["Select", "10", "20", "50", "100"],
                        key="max_iter_so2",
                    )

                # buat kolom
                if (
                    test_size != "Select"
                    and kernel != "Select"
                    and popsize != "Select"
                    and max_iter != "Select"
                ):

                    # konversi select data ke dalam bentuk numeric
                    test_size = float(test_size)
                    popsize = int(popsize)
                    max_iter = int(max_iter)

                    # pembagian data
                    training, test = train_test_split(
                        scaled_featuresX_so2,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_so2,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    st.markdown(
                        "<h2>Hasil Parameter PSO-SVR</h2>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"test size : {test_size}, kernel : {kernel}, popsize : {popsize}, max iter : {max_iter}"
                    )

                    # Define the bounds for each hyperparameter
                    if kernel == "linear":
                        lower_bound = [0.01, 0.01]  # C and epsilon only
                        upper_bound = [1, 0.1]  # C and epsilon only
                        n_dimensions = 2
                    else:
                        lower_bound = [0.01, 0.01, 0.01]
                        upper_bound = [1, 0.1, 0.1]
                        n_dimensions = 3

                    # Define the PSO algorithm
                    w = 0.5
                    c1 = 2
                    c2 = 2

                    def fitness_function(mape):
                        return 1 / (mape + 1)

                    # Define the objective function
                    # Set random seed for reproducibility
                    np.random.seed(42)

                    def objective_function(params):
                        if kernel == "linear":
                            c, epsilon = params
                            svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                        else:
                            c, gamma, epsilon = params
                            svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                        svr.fit(training, training_label)
                        pred_train = svr.predict(training)
                        pred_train1 = pred_train.reshape(-1, 1)

                        # Denormalisasi Data
                        denormalized_data_train = pd.DataFrame(
                            scaler.inverse_transform(
                                training_label.values.reshape(-1, 1)
                            ),
                            columns=["Testing Data"],
                        )
                        denormalized_data_preds = pd.DataFrame(
                            scaler.inverse_transform(pred_train1),
                            columns=["Predict Data"],
                        )
                        mape = mean_absolute_percentage_error(
                            denormalized_data_train, denormalized_data_preds
                        )
                        return mape

                    def pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    ):

                        particles = np.random.uniform(
                            low=lower_bound,
                            high=upper_bound,
                            size=(n_particles, n_dimensions),
                        )

                        personal_best_positions = particles.copy()
                        personal_best_scores = [
                            objective_function(p) for p in personal_best_positions
                        ]
                        global_best_position = particles[
                            np.argmin(personal_best_scores)
                        ]
                        global_best_error = min(personal_best_scores)
                        global_best_fitness = fitness_function(global_best_error)

                        velocities = np.zeros((n_particles, n_dimensions))

                        all_particles = []
                        all_mape = []
                        all_fitness = []
                        convergence_iter = None

                        for i in range(max_iter):

                            r1 = np.random.rand(n_particles, n_dimensions)
                            r2 = np.random.rand(n_particles, n_dimensions)
                            velocities = (
                                w * velocities
                                + c1 * r1 * (personal_best_positions - particles)
                                + c2 * r2 * (global_best_position - particles)
                            )

                            particles = particles + velocities

                            particles = np.clip(particles, lower_bound, upper_bound)

                            mape_particles = [objective_function(p) for p in particles]
                            fitness_particles = [
                                fitness_function(mape) for mape in mape_particles
                            ]
                            all_mape.append(mape_particles)
                            all_fitness.append(fitness_particles)

                            for j in range(n_particles):
                                error = mape_particles[j]
                                if error < personal_best_scores[j]:
                                    personal_best_positions[j] = particles[j]
                                    personal_best_scores[j] = error
                                    if error < global_best_error:
                                        global_best_position = particles[j]
                                        global_best_error = error
                                        global_best_fitness = fitness_function(
                                            global_best_error
                                        )

                            all_particles.append(particles.copy())

                            st.subheader(f"\nIteration {i + 1}:")
                            if kernel == "linear":
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Epsilon={global_best_position[1]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Epsilon={pbest[1]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )
                            else:
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Gamma={global_best_position[1]}, Epsilon={global_best_position[2]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Gamma={pbest[1]}, Epsilon={pbest[2]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )

                            if (
                                len(set(mape_particles)) == 1
                                and convergence_iter is None
                            ):
                                convergence_iter = i + 1

                        # Print the best particle found
                        st.markdown(
                            "<h3>Partikel Terbaik</h3>",
                            unsafe_allow_html=True,
                        )
                        if kernel == "linear":
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Epsilon: {global_best_position[1]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Gamma: {global_best_position[1]}</p>
                                    <p>Epsilon: {global_best_position[2]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        if convergence_iter is not None:
                            st.success(
                                f"\nConvergence reached at iteration {convergence_iter}"
                            )

                        return global_best_position

                    # SVR-PSO
                    # Define the PSO algorithm parameters
                    n_particles = popsize
                    max_iter = max_iter

                    # Call the PSO function with the specified hyperparameters
                    hyperparameters = pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    )

                    # Train the SVR model with the best hyperparameters and plot the results
                    if kernel == "linear":
                        c, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                    else:
                        c, gamma, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                    svr.fit(training, training_label)
                    pred_test = svr.predict(test)
                    pred_test1 = pred_test.reshape(-1, 1)

                    # Denormalisasi Data

                    test_label_array = (
                        test_label.values
                    )  # Convert DataFrame to NumPy array
                    reshaped_test_label = test_label_array.reshape(-1, 1)
                    denormalized_data_test = scaler.inverse_transform(
                        reshaped_test_label
                    )
                    denormalized_data_test = pd.DataFrame(
                        denormalized_data_test, columns=["Testing Data"]
                    )
                    denormalized_preds_test = pd.DataFrame(
                        scaler.inverse_transform(pred_test1), columns=["Predict Data"]
                    )
                    hasil_so2_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )

                    # Pengujian dengan MAPE
                    MAPE = (
                        mean_absolute_percentage_error(
                            denormalized_data_test, denormalized_preds_test
                        )
                        * 100
                    )

                    # Menampilkan hasil prediksi dan data aktual training data
                    st.subheader("Hasil Prediksi Data Uji")
                    st.write(hasil_so2_2)

                    # Plot the actual data vs predicted data
                    st.subheader("Plotting Hasil Prediksi dan Data Uji SO2")
                    chart_data_so2_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )
                    st.line_chart(chart_data_so2_2)

                    # Menampilkan Hasil Pengujian
                    st.subheader("Hasil Pengujian Data Uji")
                    st.write("MAPE")
                    st.info(f"Hasil MAPE : {MAPE}%")

        # CO
        with tabs[2]:
            st.markdown(
                "<h1 style='text-align: center; '>Modelling CO</h2>",
                unsafe_allow_html=True,
            )
            selected_co = st.selectbox(
                "Pilih Modelling", ["Select Modelling CO", "SVR CO", "SVR-PSO CO"]
            )

            # univariate CO
            X_co, y_co = split_sequence(imports_co, kolom)
            print(X_co.shape, y_co.shape)
            shapeX_co = X_co.shape
            dfX_co = pd.DataFrame(X_co)
            dfy_co = pd.DataFrame(y_co, columns=["Xt"])
            df_co = pd.concat((dfX_co, dfy_co), axis=1)

            # standardisasi co
            scaledX_co = scaler.fit_transform(dfX_co)
            scaledY_co = scaler.fit_transform(dfy_co)
            features_namesX_co = dfX_co.columns.copy()
            features_namesy_co = dfy_co.columns.copy()
            scaled_featuresX_co = pd.DataFrame(scaledX_co, columns=features_namesX_co)
            scaled_featuresY_co = pd.DataFrame(scaledY_co, columns=features_namesy_co)

            if selected_co == "SVR CO":

                # Input kernel dan test size CO
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_co",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_co",
                    )

                # Pastikan nilai test_size dan kernel valid
                if test_size != "Select" and kernel != "Select":

                    # pembagian dataset CO
                    test_size = float(test_size)  # Konversi test_size ke float
                    training, test = train_test_split(
                        scaled_featuresX_co,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_co,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    # Mengubah training_label co ke bentuk array
                    training_label = np.array(training_label).reshape(-1, 1)

                    # Membuat model SVR
                    regresor = SVR(kernel=kernel, C=1, gamma="scale", epsilon=0.1)
                    regresor.fit(training, training_label.ravel())

                    # Prediksi data testing
                    pred_test = regresor.predict(test)
                    pred_test = pred_test.reshape(-1, 1)

                    # Denormalisasi data testing
                    denormalized_train_co = pd.DataFrame(
                        scaler.inverse_transform(test_label),
                        columns=["Data Aktual CO (Testing)"],
                    )
                    denormalized_pred_co = pd.DataFrame(
                        scaler.inverse_transform(pred_test),
                        columns=["Data Prediksi CO (Testing)"],
                    )
                    hasil_co = pd.concat(
                        [denormalized_train_co, denormalized_pred_co], axis=1
                    )

                    # Hitung MAPE
                    MAPE = mean_absolute_percentage_error(
                        denormalized_train_co, denormalized_pred_co
                    )

                    # Menampilkan hasil prediksi dan data aktual test data
                    st.subheader("Prediksi Data Testing")
                    st.write(hasil_co)

                    # Menampilkan plot antara data aktual dan data prediksi pada data test
                    st.subheader("Plotting Data Aktual VS Data Prediksi - Data Testing")
                    st.line_chart(hasil_co)

                    # Menampilkan matriks evaluasi
                    st.subheader("Metriks Evaluasi Data Testing")
                    st.info(f"MAPE :\n{MAPE*100}%")

                else:
                    st.warning("Please select both a test size and a kernel.")

            elif selected_co == "SVR-PSO CO":
                # Input kernel dan test size CO
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_co",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_co",
                    )

                # Input pop size dan max iter PSO
                col1, col2 = st.columns(2)
                with col1:
                    popsize = st.selectbox(
                        "popsize",
                        ["Select", "5", "10", "20", "30"],
                        key="popsize_co",
                    )

                with col2:
                    max_iter = st.selectbox(
                        "max_iter",
                        ["Select", "10", "20", "50", "100"],
                        key="max_iter_co",
                    )

                # buat kolom
                if (
                    test_size != "Select"
                    and kernel != "Select"
                    and popsize != "Select"
                    and max_iter != "Select"
                ):

                    # konversi select data ke dalam bentuk numeric
                    test_size = float(test_size)
                    popsize = int(popsize)
                    max_iter = int(max_iter)

                    # pembagian data
                    training, test = train_test_split(
                        scaled_featuresX_co,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_co,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    st.markdown(
                        "<h2>Hasil Parameter PSO-SVR</h2>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"test size : {test_size}, kernel : {kernel}, popsize : {popsize}, max iter : {max_iter}"
                    )

                    # Define the bounds for each hyperparameter
                    if kernel == "linear":
                        lower_bound = [1, 0.01]  # C and epsilon only
                        upper_bound = [100, 0.1]  # C and epsilon only
                        n_dimensions = 2
                    else:
                        lower_bound = [1, 0.01, 0.01]
                        upper_bound = [100, 0.1, 0.1]
                        n_dimensions = 3

                    # Define the PSO algorithm
                    w = 0.5
                    c1 = 2
                    c2 = 2

                    def fitness_function(mape):
                        return 1 / (mape + 1)

                    # Define the objective function
                    # Set random seed for reproducibility
                    np.random.seed(42)

                    def objective_function(params):
                        if kernel == "linear":
                            c, epsilon = params
                            svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                        else:
                            c, gamma, epsilon = params
                            svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                        svr.fit(training, training_label)
                        pred_train = svr.predict(training)
                        pred_train1 = pred_train.reshape(-1, 1)

                        # Denormalisasi Data
                        denormalized_data_train = pd.DataFrame(
                            scaler.inverse_transform(
                                training_label.values.reshape(-1, 1)
                            ),
                            columns=["Testing Data"],
                        )
                        denormalized_data_preds = pd.DataFrame(
                            scaler.inverse_transform(pred_train1),
                            columns=["Predict Data"],
                        )
                        mape = mean_absolute_percentage_error(
                            denormalized_data_train, denormalized_data_preds
                        )
                        return mape

                    def pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    ):

                        particles = np.random.uniform(
                            low=lower_bound,
                            high=upper_bound,
                            size=(n_particles, n_dimensions),
                        )

                        personal_best_positions = particles.copy()
                        personal_best_scores = [
                            objective_function(p) for p in personal_best_positions
                        ]
                        global_best_position = particles[
                            np.argmin(personal_best_scores)
                        ]
                        global_best_error = min(personal_best_scores)
                        global_best_fitness = fitness_function(global_best_error)

                        velocities = np.zeros((n_particles, n_dimensions))

                        all_particles = []
                        all_mape = []
                        all_fitness = []
                        convergence_iter = None

                        for i in range(max_iter):

                            r1 = np.random.rand(n_particles, n_dimensions)
                            r2 = np.random.rand(n_particles, n_dimensions)
                            velocities = (
                                w * velocities
                                + c1 * r1 * (personal_best_positions - particles)
                                + c2 * r2 * (global_best_position - particles)
                            )

                            particles = particles + velocities

                            particles = np.clip(particles, lower_bound, upper_bound)

                            mape_particles = [objective_function(p) for p in particles]
                            fitness_particles = [
                                fitness_function(mape) for mape in mape_particles
                            ]
                            all_mape.append(mape_particles)
                            all_fitness.append(fitness_particles)

                            for j in range(n_particles):
                                error = mape_particles[j]
                                if error < personal_best_scores[j]:
                                    personal_best_positions[j] = particles[j]
                                    personal_best_scores[j] = error
                                    if error < global_best_error:
                                        global_best_position = particles[j]
                                        global_best_error = error
                                        global_best_fitness = fitness_function(
                                            global_best_error
                                        )

                            all_particles.append(particles.copy())

                            st.subheader(f"\nIteration {i + 1}:")
                            if kernel == "linear":
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Epsilon={global_best_position[1]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Epsilon={pbest[1]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )
                            else:
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Gamma={global_best_position[1]}, Epsilon={global_best_position[2]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Gamma={pbest[1]}, Epsilon={pbest[2]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )

                            if (
                                len(set(mape_particles)) == 1
                                and convergence_iter is None
                            ):
                                convergence_iter = i + 1

                        # Print the best particle found
                        st.markdown(
                            "<h3>Partikel Terbaik</h3>",
                            unsafe_allow_html=True,
                        )
                        if kernel == "linear":
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Epsilon: {global_best_position[1]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Gamma: {global_best_position[1]}</p>
                                    <p>Epsilon: {global_best_position[2]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        if convergence_iter is not None:
                            st.success(
                                f"\nConvergence reached at iteration {convergence_iter}"
                            )

                        return global_best_position

                    # SVR-PSO
                    # Define the PSO algorithm parameters
                    n_particles = popsize
                    max_iter = max_iter

                    # Call the PSO function with the specified hyperparameters
                    hyperparameters = pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    )

                    # Train the SVR model with the best hyperparameters and plot the results
                    if kernel == "linear":
                        c, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                    else:
                        c, gamma, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                    svr.fit(training, training_label)
                    pred_test = svr.predict(test)
                    pred_test1 = pred_test.reshape(-1, 1)

                    # Denormalisasi Data

                    test_label_array = (
                        test_label.values
                    )  # Convert DataFrame to NumPy array
                    reshaped_test_label = test_label_array.reshape(-1, 1)
                    denormalized_data_test = scaler.inverse_transform(
                        reshaped_test_label
                    )
                    denormalized_data_test = pd.DataFrame(
                        denormalized_data_test, columns=["Testing Data"]
                    )
                    denormalized_preds_test = pd.DataFrame(
                        scaler.inverse_transform(pred_test1), columns=["Predict Data"]
                    )
                    hasil_co_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )

                    # Pengujian dengan MAPE
                    MAPE = (
                        mean_absolute_percentage_error(
                            denormalized_data_test, denormalized_preds_test
                        )
                        * 100
                    )

                    # Menampilkan hasil prediksi dan data aktual training data
                    st.subheader("Hasil Prediksi Data Uji")
                    st.write(hasil_co_2)

                    # Plot the actual data vs predicted data
                    st.subheader("Plotting Hasil Prediksi dan Data Uji co")
                    chart_data_co_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )
                    st.line_chart(chart_data_co_2)

                    # Menampilkan Hasil Pengujian
                    st.subheader("Hasil Pengujian Data Uji")
                    st.write("MAPE")
                    st.info(f"Hasil MAPE : {MAPE}%")

        # NO2
        with tabs[3]:
            st.markdown(
                "<h1 style='text-align: center; '>Modelling NO2</h2>",
                unsafe_allow_html=True,
            )
            selected_no2 = st.selectbox(
                "Pilih Modelling", ["Select Modelling NO2", "SVR NO2", "SVR-PSO NO2"]
            )

            # univariate NO2
            X_no2, y_no2 = split_sequence(imports_no2, kolom)
            print(X_no2.shape, y_no2.shape)
            shapeX_no2 = X_no2.shape
            dfX_no2 = pd.DataFrame(X_no2)
            dfy_no2 = pd.DataFrame(y_no2, columns=["Xt"])
            df_no2 = pd.concat((dfX_no2, dfy_no2), axis=1)

            # standardisasi NO2
            scaledX_no2 = scaler.fit_transform(dfX_no2)
            scaledY_no2 = scaler.fit_transform(dfy_no2)
            features_namesX_no2 = dfX_no2.columns.copy()
            features_namesy_no2 = dfy_no2.columns.copy()
            scaled_featuresX_no2 = pd.DataFrame(
                scaledX_no2, columns=features_namesX_no2
            )
            scaled_featuresY_no2 = pd.DataFrame(
                scaledY_no2, columns=features_namesy_no2
            )

            if selected_no2 == "SVR NO2":

                # Input kernel dan test size NO2
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_no2",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_no2",
                    )

                # Pastikan nilai test_size dan kernel valid
                if test_size != "Select" and kernel != "Select":

                    # pembagian dataset NO2
                    test_size = float(test_size)  # Konversi test_size ke float
                    training, test = train_test_split(
                        scaled_featuresX_no2,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_no2,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    # Mengubah training_label no2 ke bentuk array
                    training_label = np.array(training_label).reshape(-1, 1)

                    # Membuat model SVR
                    regresor = SVR(kernel=kernel, C=1, gamma="scale", epsilon=0.1)
                    regresor.fit(training, training_label.ravel())

                    # Prediksi data testing
                    pred_test = regresor.predict(test)
                    pred_test = pred_test.reshape(-1, 1)

                    # Denormalisasi data testing
                    denormalized_train_no2 = pd.DataFrame(
                        scaler.inverse_transform(test_label),
                        columns=["Data Aktual NO2 (Testing)"],
                    )
                    denormalized_pred_no2 = pd.DataFrame(
                        scaler.inverse_transform(pred_test),
                        columns=["Data Prediksi NO2 (Testing)"],
                    )
                    hasil_no2 = pd.concat(
                        [denormalized_train_no2, denormalized_pred_no2], axis=1
                    )

                    # Hitung MAPE
                    MAPE = mean_absolute_percentage_error(
                        denormalized_train_no2, denormalized_pred_no2
                    )

                    # Menampilkan hasil prediksi dan data aktual test data
                    st.subheader("Prediksi Data Testing")
                    st.write(hasil_no2)

                    # Menampilkan plot antara data aktual dan data prediksi pada data test
                    st.subheader("Plotting Data Aktual VS Data Prediksi - Data Testing")
                    st.line_chart(hasil_no2)

                    # Menampilkan matriks evaluasi
                    st.subheader("Metriks Evaluasi Data Testing")
                    st.info(f"MAPE :\n{MAPE*100}%")

                else:
                    st.warning("Please select both a test size and a kernel.")

            elif selected_no2 == "SVR-PSO NO2":
                # Input kernel dan test size NO2
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_no2",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_no2",
                    )

                # Input pop size dan max iter PSO
                col1, col2 = st.columns(2)
                with col1:
                    popsize = st.selectbox(
                        "popsize",
                        ["Select", "5", "10", "20", "30"],
                        key="popsize_no2",
                    )

                with col2:
                    max_iter = st.selectbox(
                        "max_iter",
                        ["Select", "10", "20", "50", "100"],
                        key="max_iter_no2",
                    )

                # buat kolom
                if (
                    test_size != "Select"
                    and kernel != "Select"
                    and popsize != "Select"
                    and max_iter != "Select"
                ):

                    # konversi select data ke dalam bentuk numeric
                    test_size = float(test_size)
                    popsize = int(popsize)
                    max_iter = int(max_iter)

                    # pembagian data
                    training, test = train_test_split(
                        scaled_featuresX_no2,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_no2,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    st.markdown(
                        "<h2>Hasil Parameter PSO-SVR</h2>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"test size : {test_size}, kernel : {kernel}, popsize : {popsize}, max iter : {max_iter}"
                    )

                    # Define the bounds for each hyperparameter
                    if kernel == "linear":
                        lower_bound = [0.01, 0.01]  # C and epsilon only
                        upper_bound = [10, 0.1]  # C and epsilon only
                        n_dimensions = 2
                    else:
                        lower_bound = [0.01, 0.01, 0.01]
                        upper_bound = [10, 0.1, 0.1]
                        n_dimensions = 3

                    # Define the PSO algorithm
                    w = 0.5
                    c1 = 2
                    c2 = 2

                    def fitness_function(mape):
                        return 1 / (mape + 1)

                    # Define the objective function
                    # Set random seed for reproducibility
                    np.random.seed(42)

                    def objective_function(params):
                        if kernel == "linear":
                            c, epsilon = params
                            svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                        else:
                            c, gamma, epsilon = params
                            svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                        svr.fit(training, training_label)
                        pred_train = svr.predict(training)
                        pred_train1 = pred_train.reshape(-1, 1)

                        # Denormalisasi Data
                        denormalized_data_train = pd.DataFrame(
                            scaler.inverse_transform(
                                training_label.values.reshape(-1, 1)
                            ),
                            columns=["Testing Data"],
                        )
                        denormalized_data_preds = pd.DataFrame(
                            scaler.inverse_transform(pred_train1),
                            columns=["Predict Data"],
                        )
                        mape = mean_absolute_percentage_error(
                            denormalized_data_train, denormalized_data_preds
                        )
                        return mape

                    def pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    ):

                        particles = np.random.uniform(
                            low=lower_bound,
                            high=upper_bound,
                            size=(n_particles, n_dimensions),
                        )

                        personal_best_positions = particles.copy()
                        personal_best_scores = [
                            objective_function(p) for p in personal_best_positions
                        ]
                        global_best_position = particles[
                            np.argmin(personal_best_scores)
                        ]
                        global_best_error = min(personal_best_scores)
                        global_best_fitness = fitness_function(global_best_error)

                        velocities = np.zeros((n_particles, n_dimensions))

                        all_particles = []
                        all_mape = []
                        all_fitness = []
                        convergence_iter = None

                        for i in range(max_iter):

                            r1 = np.random.rand(n_particles, n_dimensions)
                            r2 = np.random.rand(n_particles, n_dimensions)
                            velocities = (
                                w * velocities
                                + c1 * r1 * (personal_best_positions - particles)
                                + c2 * r2 * (global_best_position - particles)
                            )

                            particles = particles + velocities

                            particles = np.clip(particles, lower_bound, upper_bound)

                            mape_particles = [objective_function(p) for p in particles]
                            fitness_particles = [
                                fitness_function(mape) for mape in mape_particles
                            ]
                            all_mape.append(mape_particles)
                            all_fitness.append(fitness_particles)

                            for j in range(n_particles):
                                error = mape_particles[j]
                                if error < personal_best_scores[j]:
                                    personal_best_positions[j] = particles[j]
                                    personal_best_scores[j] = error
                                    if error < global_best_error:
                                        global_best_position = particles[j]
                                        global_best_error = error
                                        global_best_fitness = fitness_function(
                                            global_best_error
                                        )

                            all_particles.append(particles.copy())

                            st.subheader(f"\nIteration {i + 1}:")
                            if kernel == "linear":
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Epsilon={global_best_position[1]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Epsilon={pbest[1]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )
                            else:
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Gamma={global_best_position[1]}, Epsilon={global_best_position[2]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Gamma={pbest[1]}, Epsilon={pbest[2]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )

                            if (
                                len(set(mape_particles)) == 1
                                and convergence_iter is None
                            ):
                                convergence_iter = i + 1

                        # Print the best particle found
                        st.markdown(
                            "<h3>Partikel Terbaik</h3>",
                            unsafe_allow_html=True,
                        )
                        if kernel == "linear":
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Epsilon: {global_best_position[1]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Gamma: {global_best_position[1]}</p>
                                    <p>Epsilon: {global_best_position[2]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        if convergence_iter is not None:
                            st.success(
                                f"\nConvergence reached at iteration {convergence_iter}"
                            )

                        return global_best_position

                    # SVR-PSO
                    # Define the PSO algorithm parameters
                    n_particles = popsize
                    max_iter = max_iter

                    # Call the PSO function with the specified hyperparameters
                    hyperparameters = pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    )

                    # Train the SVR model with the best hyperparameters and plot the results
                    if kernel == "linear":
                        c, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                    else:
                        c, gamma, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                    svr.fit(training, training_label)
                    pred_test = svr.predict(test)
                    pred_test1 = pred_test.reshape(-1, 1)

                    # Denormalisasi Data

                    test_label_array = (
                        test_label.values
                    )  # Convert DataFrame to NumPy array
                    reshaped_test_label = test_label_array.reshape(-1, 1)
                    denormalized_data_test = scaler.inverse_transform(
                        reshaped_test_label
                    )
                    denormalized_data_test = pd.DataFrame(
                        denormalized_data_test, columns=["Testing Data"]
                    )
                    denormalized_preds_test = pd.DataFrame(
                        scaler.inverse_transform(pred_test1), columns=["Predict Data"]
                    )
                    hasil_no2_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )

                    # Pengujian dengan MAPE
                    MAPE = (
                        mean_absolute_percentage_error(
                            denormalized_data_test, denormalized_preds_test
                        )
                        * 100
                    )

                    # Menampilkan hasil prediksi dan data aktual training data
                    st.subheader("Hasil Prediksi Data Uji")
                    st.write(hasil_no2_2)

                    # Plot the actual data vs predicted data
                    st.subheader("Plotting Hasil Prediksi dan Data Uji NO2")
                    chart_data_no2_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )
                    st.line_chart(chart_data_no2_2)

                    # Menampilkan Hasil Pengujian
                    st.subheader("Hasil Pengujian Data Uji")
                    st.write("MAPE")

        # O3
        with tabs[4]:
            st.markdown(
                "<h1 style='text-align: center; '>Modelling O3</h2>",
                unsafe_allow_html=True,
            )
            selected_o3 = st.selectbox(
                "Pilih Modelling", ["Select Modelling O3", "SVR O3", "SVR-PSO O3"]
            )

            # univariate O3
            X_o3, y_o3 = split_sequence(imports_o3, kolom)
            shapeX_o3 = X_o3.shape
            dfX_o3 = pd.DataFrame(X_o3)
            dfy_o3 = pd.DataFrame(y_o3, columns=["Xt"])
            df_o3 = pd.concat((dfX_o3, dfy_o3), axis=1)

            # standardisasi O3
            scaledX_o3 = scaler.fit_transform(dfX_o3)
            scaledY_o3 = scaler.fit_transform(dfy_o3)
            features_namesX_o3 = dfX_o3.columns.copy()
            features_namesy_o3 = dfy_o3.columns.copy()
            scaled_featuresX_o3 = pd.DataFrame(scaledX_o3, columns=features_namesX_o3)
            scaled_featuresY_o3 = pd.DataFrame(scaledY_o3, columns=features_namesy_o3)

            if selected_o3 == "SVR O3":

                # Input kernel dan test size O3
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_o3",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_o3",
                    )

                # Pastikan nilai test_size dan kernel valid
                if test_size != "Select" and kernel != "Select":

                    # pembagian dataset O3
                    test_size = float(test_size)  # Konversi test_size ke float
                    training, test = train_test_split(
                        scaled_featuresX_o3,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_o3,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    # Mengubah training_label O3 ke bentuk array
                    training_label = np.array(training_label).reshape(-1, 1)

                    # Membuat model SVR
                    regresor = SVR(kernel=kernel, C=1, gamma="scale", epsilon=0.1)
                    regresor.fit(training, training_label.ravel())

                    # Prediksi data testing
                    pred_test = regresor.predict(test)
                    pred_test = pred_test.reshape(-1, 1)

                    # Denormalisasi data testing
                    denormalized_train_o3 = pd.DataFrame(
                        scaler.inverse_transform(test_label),
                        columns=["Data Aktual O3 (Testing)"],
                    )
                    denormalized_pred_o3 = pd.DataFrame(
                        scaler.inverse_transform(pred_test),
                        columns=["Data Prediksi O3 (Testing)"],
                    )
                    hasil_o3 = pd.concat(
                        [denormalized_train_o3, denormalized_pred_o3], axis=1
                    )

                    # Hitung MAPE
                    MAPE = mean_absolute_percentage_error(
                        denormalized_train_o3, denormalized_pred_o3
                    )

                    # Menampilkan hasil prediksi dan data aktual test data
                    st.subheader("Prediksi Data Testing")
                    st.write(hasil_o3)

                    # Menampilkan plot antara data aktual dan data prediksi pada data test
                    st.subheader("Plotting Data Aktual VS Data Prediksi - Data Testing")
                    st.line_chart(hasil_o3)

                    # Menampilkan matriks evaluasi
                    st.subheader("Metriks Evaluasi Data Testing")
                    st.info(f"MAPE :\n{MAPE*100}%")

                else:
                    st.warning("Please select both a test size and a kernel.")

            elif selected_o3 == "SVR-PSO O3":
                # Input kernel dan test size O3
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.selectbox(
                        "Tes Size",
                        ["Select", "0.1", "0.2", "0.3", "0.4", "0.5"],
                        key="test_size_o3",
                    )

                with col2:
                    kernel = st.selectbox(
                        "Kernel",
                        ["Select", "rbf", "linear", "poly"],
                        key="kernel_o3",
                    )

                # Input pop size dan max iter PSO
                col1, col2 = st.columns(2)
                with col1:
                    popsize = st.selectbox(
                        "popsize",
                        ["Select", "5", "10", "20", "30"],
                        key="popsize_o3",
                    )

                with col2:
                    max_iter = st.selectbox(
                        "max_iter",
                        ["Select", "10", "20", "50", "100"],
                        key="max_iter_o3",
                    )

                # buat kolom
                if (
                    test_size != "Select"
                    and kernel != "Select"
                    and popsize != "Select"
                    and max_iter != "Select"
                ):

                    # konversi select data ke dalam bentuk numeric
                    test_size = float(test_size)
                    popsize = int(popsize)
                    max_iter = int(max_iter)

                    # pembagian data
                    training, test = train_test_split(
                        scaled_featuresX_o3,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )
                    training_label, test_label = train_test_split(
                        scaled_featuresY_o3,
                        test_size=test_size,
                        random_state=0,
                        shuffle=False,
                    )

                    st.markdown(
                        "<h2>Hasil Parameter PSO-SVR</h2>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"test size : {test_size}, kernel : {kernel}, popsize : {popsize}, max iter : {max_iter}"
                    )

                    # Define the bounds for each hyperparameter
                    if kernel == "linear":
                        lower_bound = [0.01, 0.01]  # C and epsilon only
                        upper_bound = [10, 0.1]  # C and epsilon only
                        n_dimensions = 2
                    else:
                        lower_bound = [0.01, 0.01, 0.01]
                        upper_bound = [10, 0.1, 0.1]
                        n_dimensions = 3

                    # Define the PSO algorithm
                    w = 0.5
                    c1 = 2
                    c2 = 2

                    def fitness_function(mape):
                        return 1 / (mape + 1)

                    # Define the objective function
                    # Set random seed for reproducibility
                    np.random.seed(42)

                    def objective_function(params):
                        if kernel == "linear":
                            c, epsilon = params
                            svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                        else:
                            c, gamma, epsilon = params
                            svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                        svr.fit(training, training_label)
                        pred_train = svr.predict(training)
                        pred_train1 = pred_train.reshape(-1, 1)

                        # Denormalisasi Data
                        denormalized_data_train = pd.DataFrame(
                            scaler.inverse_transform(
                                training_label.values.reshape(-1, 1)
                            ),
                            columns=["Testing Data"],
                        )
                        denormalized_data_preds = pd.DataFrame(
                            scaler.inverse_transform(pred_train1),
                            columns=["Predict Data"],
                        )
                        mape = mean_absolute_percentage_error(
                            denormalized_data_train, denormalized_data_preds
                        )
                        return mape

                    def pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    ):

                        particles = np.random.uniform(
                            low=lower_bound,
                            high=upper_bound,
                            size=(n_particles, n_dimensions),
                        )

                        personal_best_positions = particles.copy()
                        personal_best_scores = [
                            objective_function(p) for p in personal_best_positions
                        ]
                        global_best_position = particles[
                            np.argmin(personal_best_scores)
                        ]
                        global_best_error = min(personal_best_scores)
                        global_best_fitness = fitness_function(global_best_error)

                        velocities = np.zeros((n_particles, n_dimensions))

                        all_particles = []
                        all_mape = []
                        all_fitness = []
                        convergence_iter = None

                        for i in range(max_iter):

                            r1 = np.random.rand(n_particles, n_dimensions)
                            r2 = np.random.rand(n_particles, n_dimensions)
                            velocities = (
                                w * velocities
                                + c1 * r1 * (personal_best_positions - particles)
                                + c2 * r2 * (global_best_position - particles)
                            )

                            particles = particles + velocities

                            particles = np.clip(particles, lower_bound, upper_bound)

                            mape_particles = [objective_function(p) for p in particles]
                            fitness_particles = [
                                fitness_function(mape) for mape in mape_particles
                            ]
                            all_mape.append(mape_particles)
                            all_fitness.append(fitness_particles)

                            for j in range(n_particles):
                                error = mape_particles[j]
                                if error < personal_best_scores[j]:
                                    personal_best_positions[j] = particles[j]
                                    personal_best_scores[j] = error
                                    if error < global_best_error:
                                        global_best_position = particles[j]
                                        global_best_error = error
                                        global_best_fitness = fitness_function(
                                            global_best_error
                                        )

                            all_particles.append(particles.copy())

                            st.subheader(f"\nIteration {i + 1}:")
                            if kernel == "linear":
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Epsilon={global_best_position[1]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Epsilon={pbest[1]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )
                            else:
                                st.write(
                                    f"Global Best Position (gbest): C={global_best_position[0]}, Gamma={global_best_position[1]}, Epsilon={global_best_position[2]}, Global Best MAPE: {global_best_error*100}%, Global Best Fitness: {global_best_fitness}"
                                )
                                for p_idx, (pbest, pbest_score, fit) in enumerate(
                                    zip(
                                        personal_best_positions,
                                        personal_best_scores,
                                        fitness_particles,
                                    )
                                ):
                                    st.markdown(
                                        f"**Particle {p_idx + 1}**\n\n"
                                        f"Personal Best Position (pbest): C={pbest[0]}, Gamma={pbest[1]}, Epsilon={pbest[2]}\n\n"
                                        f"Personal Best MAPE: {pbest_score*100}% \n\n"
                                        f"\nFitness: {fit}"
                                    )

                            if (
                                len(set(mape_particles)) == 1
                                and convergence_iter is None
                            ):
                                convergence_iter = i + 1

                        # Print the best particle found
                        st.markdown(
                            "<h3>Partikel Terbaik</h3>",
                            unsafe_allow_html=True,
                        )
                        if kernel == "linear":
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Epsilon: {global_best_position[1]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="border: 1px solid #D3D3D3; padding: 10px; border-radius: 5px;">
                                    <p><strong>Best Particle atau Parameter:</strong></p>
                                    <p>C: {global_best_position[0]}</p>
                                    <p>Gamma: {global_best_position[1]}</p>
                                    <p>Epsilon: {global_best_position[2]}</p>
                                    <br>
                                    <p><strong>Best MAPE:</strong></p>
                                    <p>{global_best_error*100}%</p>
                                    <br>
                                    <p><strong>Best Fitness:</strong></p>
                                    <p>{global_best_fitness}</p>
                                </div><br>
                                """,
                                unsafe_allow_html=True,
                            )
                        if convergence_iter is not None:
                            st.success(
                                f"\nConvergence reached at iteration {convergence_iter}"
                            )

                        return global_best_position

                    # SVR-PSO
                    # Define the PSO algorithm parameters
                    n_particles = popsize
                    max_iter = max_iter

                    # Call the PSO function with the specified hyperparameters
                    hyperparameters = pso(
                        objective_function,
                        lower_bound,
                        upper_bound,
                        n_particles,
                        n_dimensions,
                        max_iter,
                        w,
                        c1,
                        c2,
                    )

                    # Train the SVR model with the best hyperparameters and plot the results
                    if kernel == "linear":
                        c, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, epsilon=epsilon)
                    else:
                        c, gamma, epsilon = hyperparameters
                        svr = SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon)

                    svr.fit(training, training_label)
                    pred_test = svr.predict(test)
                    pred_test1 = pred_test.reshape(-1, 1)

                    # Denormalisasi Data

                    test_label_array = (
                        test_label.values
                    )  # Convert DataFrame to NumPy array
                    reshaped_test_label = test_label_array.reshape(-1, 1)
                    denormalized_data_test = scaler.inverse_transform(
                        reshaped_test_label
                    )
                    denormalized_data_test = pd.DataFrame(
                        denormalized_data_test, columns=["Testing Data"]
                    )
                    denormalized_preds_test = pd.DataFrame(
                        scaler.inverse_transform(pred_test1), columns=["Predict Data"]
                    )
                    hasil_o3_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )

                    # Pengujian dengan MAPE
                    MAPE = (
                        mean_absolute_percentage_error(
                            denormalized_data_test, denormalized_preds_test
                        )
                        * 100
                    )

                    # Menampilkan hasil prediksi dan data aktual training data
                    st.subheader("Hasil Prediksi Data Uji")
                    st.write(hasil_o3_2)

                    # Plot the actual data vs predicted data
                    st.subheader("Plotting Hasil Prediksi dan Data Uji O3")
                    chart_data_o3_2 = pd.concat(
                        [denormalized_data_test, denormalized_preds_test], axis=1
                    )
                    st.line_chart(chart_data_o3_2)

                    # Menampilkan Hasil Pengujian
                    st.subheader("Hasil Pengujian Data Uji")
                    st.write("MAPE")
                    st.info(f"Hasil MAPE : {MAPE}%")

    elif selected == "Best Parameters":

        import warnings

        warnings.filterwarnings("ignore")

        st.subheader("Parameter Terbaik dari Setiap Polutan")

        # Data untuk tabel
        best_params_so2 = {
            "Kernel": ["Linear", "RBF", "Polynomial"],
            "C": [1, 1, 0.87818989804329],
            "Gamma": ["-", 0.1, 0.0339697407485143],
            "Epsilon": [0.0108744826595038, 0.1, 0.0275822826417232],
            "MAPE (%)": [5.53534472485894, 5.35379482278262, 24.4738266284509],
        }
        best_params_pm10 = {
            "Kernel": ["Linear", "RBF", "Polynomial"],
            "C": [11.0544692990733, 10, 5.16054074436355],
            "Gamma": ["-", 0.1, 0.1],
            "Epsilon": [0.0336424227720405, 0.0100768077777405, 0.1],
            "MAPE (%)": [14.9544, 13.92424, 21.11832],
        }
        best_params_co = {
            "Kernel": ["Linear", "RBF", "Polynomial"],
            "C": [17.28203588, 100, 0.753307221109347],
            "Gamma": ["-", 0.1, 0.1],
            "Epsilon": [0.0206381067757196, 0.01, 0.01],
            "MAPE (%)": [26.32783429, 25.0772342222504, 55.1845017653986],
        }
        best_params_o3 = {
            "Kernel": ["Linear", "RBF", "Polynomial"],
            "C": [3.721195616, 10, 10],
            "Gamma": ["-", 0.1, 0.1],
            "Epsilon": [0.01, 0.01, 0.0143638214921914],
            "MAPE (%)": [13.744498347988, 14.6582739710206, 43.408241131151],
        }
        best_params_no2 = {
            "Kernel": ["Linear", "RBF", "Polynomial"],
            "C": [11.0544692990733, 10, 5.16054074436355],
            "Gamma": ["-", 0.1, 0.1],
            "Epsilon": [0.0336424227720405, 0.0100768077777405, 0.1],
            "MAPE (%)": [23.31841, 25.45876, 44.98468],
        }

        # Membuat DataFrame dari data
        df_pm10 = pd.DataFrame(best_params_pm10)
        df_so2 = pd.DataFrame(best_params_so2)
        df_co = pd.DataFrame(best_params_co)
        df_o3 = pd.DataFrame(best_params_o3)
        df_no2 = pd.DataFrame(best_params_no2)

        # Membuat expander
        with st.expander("PM10"):
            st.dataframe(df_pm10)
        with st.expander("SO2"):
            st.dataframe(df_so2)
        with st.expander("CO"):
            st.dataframe(df_co)
        with st.expander("O3"):
            st.dataframe(df_o3)
        with st.expander("NO2"):
            st.dataframe(df_no2)

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
