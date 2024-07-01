import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from io import BytesIO

def kmeans_analysis(df, n_clusters):
    # Convert 'total nilai' column to float
    df['total nilai'] = df['total nilai'].str.replace(',', '.').astype(float)

    # Data preparation
    x_train = df[['uts', 'uas', 'total nilai']].values
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)  # Apply scaling

    # Ensure x_train_scaled is float64
    x_train_scaled = x_train_scaled.astype(np.float64)

    # K-Means clustering
    kmean = KMeans(n_clusters=n_clusters)
    kmean.fit(x_train_scaled)
    y_cluster = kmean.predict(x_train_scaled)  # Use predict to get cluster labels

    # Cluster assignment
    df['Cluster'] = y_cluster

    # Create cluster names based on average score
    cluster_names = ['Kelas C' if (val >= 70 and val <= 79) else
                     ('Kelas B' if (val >= 80 and val <=85) else 'Kelas Unggulan')
                     for val in df['total nilai']]
    df['Cluster Name'] = cluster_names

    # Prepare data for bar plot
    cluster_counts = df['Cluster Name'].value_counts()

    # Bar plot
    fig_bar = plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Jumlah Siswa Setiap Kelas')
    plt.ylabel('Jumlah Siswa')
    plt.title('Distribusi Siswa Pada Pembagian Kelas')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Scatter plot of uploaded data
    fig_scatter = plt.figure(figsize=(10, 6))
    plt.scatter(df['uts'], df['uas'], c=df['Cluster'], cmap='viridis', alpha=0.6)
    plt.xlabel('Nilai UTS')
    plt.ylabel('Nilai UAS')
    plt.title('Plot Sebaran Nilai UTS dan UAS Siswa')
    plt.colorbar(label='Cluster')
    plt.tight_layout()

    return df, fig_bar, fig_scatter

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# Streamlit app
with st.sidebar:
    selected = option_menu('Aplikasi K-means Clustering Pembagian Kelas Unggulan SDN IPK Ciriung 01',['Tentang Sekolah','Tentang K-Means Cluster','Hitung Pembagian Kelas Unggulan'], 
                           icons=['info-square','book','clipboard2-data'], menu_icon="menu-button-wide", default_index=0)

if selected == 'Hitung Pembagian Kelas Unggulan':
    st.title("K-Means Clustering Untuk Menganalisa Pembagian Kelas Unggulan Pada Sekolah SDN IPK Ciriung 01")

    # Upload CSV data
    uploaded_file = st.file_uploader("Upload file data siswa berbentuk CSV (uts, uas, total nilai)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=";")

        # Input for number of clusters with informative label
        n_clusters = st.number_input("Number of Clusters (recommended range: 2-5)", min_value=1, max_value=10)

        # Button to trigger analysis
        if st.button("Perform K-Means Clustering"):
            df, fig_bar, fig_scatter = kmeans_analysis(df.copy(), n_clusters)

            # Display descriptive statistics
            st.header("Deskripsi Statistik Dataset")
            st.write(df.describe())

            # Display data with cluster assignments based on ranges
            st.header("Pembagian Kelas Siswa")
            st.dataframe(df)

            # Download button for the Excel file
            st.download_button(
                label="Download File Excel",
                data=to_excel(df),
                file_name='pembagian_kelas_unggulan.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            # Display K-Means clustering bar plot
            st.header("Memvisualisasikan K-Means Clustering (Bar Plot)")
            st.pyplot(fig_bar)

            # Display Scatter plot
            st.header("Plot Sebaran UTS dan UAS Menggunakan Clusters")
            st.pyplot(fig_scatter)

    else:
        st.info("Silahkan Masukan File dengan Format CSV")

if selected == 'Tentang K-Means Cluster':
    st.title('Tentang K-Means Cluster')

    st.write("""
    ## Apa itu K-Means Clustering?
    K-Means Clustering adalah salah satu algoritma clustering yang paling populer dan sering digunakan dalam analisis data. Algoritma ini bertujuan untuk membagi dataset ke dalam beberapa kelompok (cluster) yang berbeda, di mana data dalam setiap cluster memiliki karakteristik yang mirip satu sama lain dan berbeda dari data di cluster lain.

    ### Bagaimana K-Means Clustering Bekerja?
    Proses K-Means Clustering dapat dijelaskan dalam beberapa langkah sebagai berikut:
    1. **Menentukan Jumlah Cluster (K)**: Sebagai langkah awal, kita harus menentukan berapa jumlah cluster (K) yang diinginkan.
    2. **Menginisialisasi Centroid**: Pilih secara acak K titik dalam dataset sebagai titik pusat awal (centroid) untuk setiap cluster.
    3. **Mengalokasikan Setiap Titik ke Cluster Terdekat**: Setiap titik data diukur jaraknya ke setiap centroid dan dialokasikan ke cluster dengan centroid terdekat.
    4. **Mengupdate Centroid**: Setelah semua titik data dialokasikan ke cluster, hitung ulang posisi centroid sebagai rata-rata dari semua titik data dalam cluster tersebut.
    5. **Iterasi**: Ulangi langkah 3 dan 4 sampai centroid tidak lagi berubah secara signifikan atau sampai jumlah iterasi yang telah ditentukan tercapai.

    ### Kelebihan K-Means Clustering:
    - **Sederhana dan Cepat**: Algoritma ini relatif mudah diimplementasikan dan cepat dalam komputasi, terutama untuk dataset besar.
    - **Skalabilitas**: K-Means dapat dengan mudah diskalakan untuk menangani dataset yang sangat besar.

    ### Kelemahan K-Means Clustering:
    - **Pemilihan Jumlah Cluster (K)**: Algoritma ini memerlukan penentuan jumlah cluster (K) di awal, yang mungkin tidak selalu jelas.
    - **Sensitif Terhadap Inisialisasi Centroid**: Hasil clustering dapat bervariasi berdasarkan pemilihan centroid awal.
    - **Tidak Efektif untuk Bentuk Cluster Non-Bulat**: K-Means bekerja terbaik ketika cluster memiliki bentuk bulat dengan ukuran yang serupa.

    ### Aplikasi K-Means Clustering:
    K-Means Clustering digunakan dalam berbagai aplikasi, seperti:
    - **Segmentasi Pelanggan**: Mengelompokkan pelanggan berdasarkan perilaku pembelian mereka.
    - **Pengelompokan Dokumen**: Mengelompokkan dokumen berdasarkan topik atau isi.
    - **Pengelompokan Gambar**: Mengelompokkan gambar berdasarkan kesamaan visual.

    K-Means Clustering adalah alat yang kuat dalam analisis data yang dapat membantu menemukan pola dan struktur tersembunyi dalam dataset.
    """)

if selected == 'Tentang Sekolah':
    st.title('Tentang Sekolah')
    
    # Display the school image
    st.image("https://lh3.googleusercontent.com/p/AF1QipMiVMkEJhVMFQrDLnVNUYGBu49Rslob35SB6FT1=s1360-w1360-h1020", caption="SDN IPK Ciriung 01", use_column_width=True)

    st.write("""
    ## Sejarah Sekolah SDN IPK Ciriung 01

    **Sekolah Dasar Negeri IPK Ciriung 01** merupakan salah satu institusi pendidikan dasar yang terletak di Jl.mayor Oking Jaya Atmaja, CIRIUNG, Kec. Cibinong, Kab. Bogor Prov. Jawa Barat.

    ### Visi dan Misi
    SDN IPK Ciriung 01 memiliki visi untuk menjadi sekolah dasar unggulan yang menghasilkan lulusan yang berprestasi, berakhlak mulia, dan mampu bersaing di era globalisasi. Misi sekolah ini meliputi:
    - Memberikan pendidikan yang berkualitas dan merata bagi seluruh siswa.
    - Menumbuhkan nilai-nilai moral dan etika dalam proses pendidikan.
    - Mengembangkan potensi siswa dalam bidang akademik dan non-akademik.
    - Meningkatkan profesionalisme guru dan tenaga kependidikan.

    ### Masa Depan
    Dengan komitmen yang kuat terhadap pendidikan berkualitas, SDN IPK Ciriung 01 terus berusaha untuk meningkatkan mutu pendidikan dan fasilitas sekolah. Harapan ke depannya adalah untuk terus menghasilkan generasi penerus bangsa yang cerdas, berkarakter, dan siap menghadapi tantangan zaman.

    SDN IPK Ciriung 01 bangga menjadi bagian dari perjalanan pendidikan anak-anak di Ciriung dan sekitarnya, dan bertekad untuk terus memberikan yang terbaik bagi siswa-siswinya.
    """)
   
