# Laporan Proyek Machine Learning - Hardianto Tandi Seno

## Domain Proyek
Pasar saham adalah salah satu instrumen investasi yang paling dinamis dan dipengaruhi oleh banyak faktor internal dan eksternal perusahaan, seperti kebijakan pemerintah, sentimen pasar, dan kondisi ekonomi. Memprediksi harga saham menjadi tantangan yang menarik sekaligus penting bagi investor, analis, dan pelaku pasar lainnya. Sebagai salah satu perusahaan telekomunikasi terbesar di Indonesia, PT Indosat Tbk (ISAT) sering menjadi perhatian pelaku pasar karena sahamnya yang aktif diperdagangkan. ISAT adalah pilihan yang ideal untuk penelitian prediksi harga saham berbasis data historis karena memiliki volume perdagangan yang besar dan volatilitas harga yang tinggi.

Pendekatan time-series atau deret waktu telah banyak digunakan untuk menganalisis dan memprediksi harga saham dalam beberapa tahun terakhir. Karena keakuratannya dalam mengidentifikasi pola musiman dan tren, model konvensional seperti ARIMA masih banyak digunakan. Namun, model deep learning seperti LSTM (Long Short-Term Memory) & GRU (Gated Recurrent Unit) mulai menunjukkan keunggulan dalam menangani kompleksitas data keuangan seiring dengan kemajuan teknologi dan ketersediaan data yang lebih besar.Dari pendekatan tersebut, tujuan dari proyek ini adalah untuk membangun sistem prediksi harga saham Indosat menggunakan pendekatan time-series dan membandingkan kinerja beberapa model prediktif, baik statistik maupun berbasis ML. Diharapkan proyek ini dapat memberikan gambaran mengenai model mana yang paling efektif dalam memprediksi pergerakan harga saham ISAT dan memberikan gambaran tentang pola pergerakan harga saham tersebut.

Beberapa penelitian terkait telah mencoba membandingkan beberapa model (ARIMA, LSTM, dan GRU) dalam memprediksi harga saham. Pada penelitian [1], peneliti mencoba untuk melakukan prediksi pada 50 saham perusahaan yang terdaftar di finance.yahoo.com dengan ARIMA & LSTM. Hasil akhirnya yaitu model ARIMA dan LSTM dapat digunakan untuk memprediksi harga saham dikarenakan prediksi yang sebagian besar konsisten dengan hasil yang sebenarnya meskipun LSTM lebih baik dalam memprediksi harga, terutama menggambarkan perubahan harga. Penelitian berikutnya [2] mencoba membandingkan RNN, LSTM, dan GRU dalam memprediksi harga saham Honda Motor Company, Oracle, dan Intuit Incorporation (sumbernya dari finance.yahoo.com). Penelitian tersebut menghasilkan kesimpulan akhir model GRU lebih menghasilkan nilai kesalahan yang lebih rendah dibandingkan dengan LSTM dan RNN.

Dengan menggunakan kedua penelitian tersebut sebagai sumber kredibel, diharapkan proyek ini bisa membandingkan dan menentukan model yang memiliki performa baik untuk beberapa jenis model yang akan diterapkan (ARIMA, LSTM, dan GRU) pada data saham Indosat.

## Business Understanding

Tujuan dari proyek ini adalah untuk membuat model prediksi harga saham PT Indosat Tbk (ISAT) menggunakan beberapa model (ARIMA, LSTM, & GRU). Model ini akan dilatih dengan menggunakan data historis harga saham Indosat yang diperoleh dari Yahoo Finance. Kemudian, data uji akan digunakan untuk memprediksi pergerakan harga saham pada periode mendatang.

Perbandingan beberapa model ini untuk melihat kecocokan dataset terhadap model tertentu. Selain itu, pengembangan model ini juga membuat investor dapat memperoleh gambaran yang lebih jelas tentang tren harga saham Indosat di masa depan dengan menggunakan model prediktif yang dapat diandalkan. Ini membantu mereka memiliki kemampuan untuk membuat strategi investasi yang lebih baik, mengoptimalkan potensi keuntungan, dan meningkatkan keyakinan dalam membuat keputusan finansial. Hal ini tidak hanya membantu mencapai tujuan investasi, tetapi juga meningkatkan keyakinan mereka dalam menghadapi dinamika pasar saham.

### Problem Statements
- Bagaimana investor dapat membuat keputusan investasi yang lebih baik secara akurat?
- Bagaimana performa model ARIMA, LSTM, dan GRU saat diterapkan dengan harga aktual pada periode pengujian, dan apakah model ini dapat digunakan sebagai alat bantu dalam pengambilan keputusan investasi?

### Goals
- Membantu investor menemukan cara yang tepat dalam membuat keputusan investasi yang lebih baik secara akurat
- Mengembangan dan membandingan beberapa model (ARIMA, LSTM, dan GRU) saat diterapkan pada periode pengujian dalam menilai 

### Solution statements
- Dalam mencapai tujuan ini, akan digunakan 3 jenis pemodelan (ARIMA, LSTM, dan GRU) untuk mencoba membandingkan performa terbaik yang dapat diterapkan dalam dataset harga saham dari perusahaan Indosat
- Terdapat opsi melakukan Normalisasi/Standarisasi (tergantung pada bentuk distribusi kolom) dan Transformasi jika menggunakan model konvensional (ARIMA) melihat hasil p-value-nya nanti.
- Pengukuran performa setiap model akan menggunakan MSE (Mean Squared Error) dan MAE (Mean Absolute Error)  

## Data Understanding
Dataset yang digunakan yaitu data historis harian saham Indosat (ISAT.JK) yang diambil pada Yahoo Finance (https://finance.yahoo.com/quote/ISAT.JK/). Datasetnya diambil menggunakan bantuan library pada Python (yfinance) dengan rentang tahun 2018-06 hingga 2025-04. Dataset terdiri dari 1679 baris dan 6 kolom.

### Variabel-variabel pada dataset tersebut adalah sebagai berikut:
- Date = Tanggal dan waktu transaksi saham
- Close = Harga saham saat pasar ditutup pada akhir sesi perdagangan
- High = Harga tertinggi yang dicapai saham selama sesi perdagangan hari itu.
- Low = Harga terendah yang dicapai saham selama hari itu.
- Open = Harga saham ketika pasar pertama kali dibuka pada awal sesi perdagangan
- Volume = Total volume yang diperdagangkan dalam rentang waktu tersebut

### Tahapan Pemahaman pada Data:
- Diawali dengan menggali infomasi pada dataset (jumlah baris-kolom, tipe data setiap kolom, mengecek statistik deskriptifnya, identifikasi missing value & outlier, melihat visualisasi distribusi & hubungan pola dalam data, dll.)
- Beberapa Explorartory Data Analysis juga dilakukan dalam rangka mencoba memahami dataset yang ada, seperti melihat tren data dari waktu ke waktu, mengecek seasonal compose pada beberapa kolom, dan melihat grafik ACF dan PACF pada setiap kolomnya.

## Data Preparation
Secara singkat, dataset historis harga saham diambil menggunakan bantuan library yfinance pada Python, kemudian dilakukan sedikit penyesuaian pada hasil pengambilan dataset tersebut lalu disimpan dalam bentuk csv. Dataset kemudian di-load kembali dengan menggunakan pandas. Alur berikutnya yaitu melakukan Data Understanding dan EDA.

Setelah itu, masuk dalam tahapan Data Preparation. Tahapan ini diawali dengan proses Standarisasi pada kolom-kolom data setelah melihat sebaran distribusinya melalui visualisasi pada bagian sebelumnya. Ini akan menyetarakan skala semua fitur yang berefek pada proses training pada model yang akan berlangsung secara lebih stabil dan efisien.

Mengubah kolom 'Date' menjadi index pada dataset juga dilakukan pada bagian ini untuk memastikan bahwa dataset ini berurutan berdasarkan waktu. Hal ini dikarenakan model time-series sangat bergantung pada urutan waktu. Jika ini tidak dilakukan, maka hanya dianggap kolom biasa, bukan sebagai acuan waktu dan model atau analisis time-series bisa salah atau jadi tidak akurat karena kehilangan konteks waktu.

Terdapat juga penyesuaian dalam proses train-test split. Penyesuaian dilakukan sesuai dengan jenis pemodelan yang akan diterapkan.
  - Untuk model ARIMA, tahapan proses train-test split diawali dengan pengecekan stationaritas data untuk menentukan perlunya tindakan differencing/tidak. Setelah itu, proses train-test split dilakukan dengan perbandingan 70%:30%.
  - Untuk model Deep Learning (LSTM & GRU), tahapannya bermula dengan pembuatan fungsi windowed agar dataset yang akan digunakan bisa diterapkan dalam arsitektur model Deep Learning nantinya. Kemudian, proses train-test split dilakukan dengan perbandingan 70%:30% dan penerapan fungsi windowed terhadap hasil split dataset.
  - Kolom data yang digunakan dalam proses pemodelan ini yaitu kolom 'Closed' (Univariate)
  - Secara keseluruhan proses train-test split data ini perlu dilakukan untuk memastikan performa model bisa diukur dengan membandingkan hasil prediksi dan hasil aktual dari dataset selain train untuk melihat kemampuan generalisasi model. 

## Modelling
Dalam proyek ini, dicoba 3 jenis model yang akan melalui tahap pealtihan dan pengujian untuk nantinya dibandingkan satu dengan yang lainnya.
  - Model ARIMA diterapkan dengan bantuan library statsmodel dengan parameter yang ditetapkan yaitu order=(1,1,2). Beberapa kelebihan dari model ini yaitu mampu mengakomodasi komponen musiman dalam data, efektif untuk memodelkan data stasioner musiman dengan noise rendah hingga sedang, dan bisa bekerja cukup baik meskipun dengan dataset yang relatif kecil. Sedangkan kelemahannya yaitu hanya bekerja optimal jika data sudah stasioner, kurang fleksibel untuk data kompleks, hanya bisa memodelkan satu model saja (univariat saja).
  - Untuk model LSTM & GRU, penerapannya dilakukan dengan menggunakan TensorFlow. Dalam penerapannya, terdapat 3 layer tambahan yang digunakan, seperti BatchNormalization & Dropout (menstabilkan proses pembelajaran dan meningkatkan generalisasi model), dan Dense (menghasilkan output akhir (harga saham)). Optimizer yang digunakan dalam pemodelan ini yaitu Adam dengan learning_rate = 0.001 karena berkaitan dengan kemampuan untuk cepat belajar, adaptif terhadap data, stabil untuk data fluktuatif seperti saham,praktis dan efektif untuk model LSTM/GRU. Selain itu, terdapat juga penerapan callback (fungsi atau objek yang dipanggil secara otomatis selama proses pelatihan (training), baik di akhir epoch, batch, atau saat training selesai) dengan berfokus pada 3 fungsi, yaitu Early Stopping (Menghentikan training lebih awal jika tidak ada peningkatan performa pada data validasi setelah beberapa epoch), ReduceLROnPlateau (Mengurangi learning rate saat model stagnan (loss tidak turun)), dan ModelCheckpoint (Menyimpan bobot model terbaik selama training (biasanya saat val_loss paling kecil)). Epoch yang ditetapkan untuk kedua model ini juga sama, yaitu 50.
  - LSTM (Long Short-Term Memory) unggul dalam mempelajari pola jangka panjang dan menangani data time-series yang bersifat non-linear seperti harga saham. Ia mampu mengingat informasi penting dari waktu ke waktu dan mengabaikan noise dengan mekanisme gating-nya. Model ini sangat cocok untuk data yang memiliki dependensi waktu, meskipun pelatihannya bisa lebih lambat dan butuh resource komputasi lebih besar dibanding model statistik seperti ARIMA. Kekurangannya, LSTM rawan overfitting jika data sedikit, dan hasilnya cenderung sulit diinterpretasi secara statistik.
  - GRU (Gated Recurrent Unit) adalah versi yang lebih sederhana dan ringan dari LSTM, namun tetap mampu menangani dependensi jangka panjang dalam data time-series. GRU memiliki arsitektur yang lebih ringkas (tanpa cell state), sehingga lebih cepat dilatih dan efisien secara komputasi. Meski lebih ringan, performa GRU seringkali setara atau mendekati LSTM, terutama saat data tidak terlalu kompleks. Namun, karena tidak sekompleks LSTM, GRU bisa kurang fleksibel dalam menangani pola-pola waktu yang sangat panjang dan rumit. Seperti LSTM, GRU juga kurang transparan secara interpretasi dibanding model statistik.

Model terbaik akan dipilih berdasarkan kinerja model selama tahap pelatihan dan evaluasi metrik dari setiap model yang telah dilatih. Pemilihan ini bertujuan untuk memperoleh model yang tidak hanya unggul secara metrik, tetapi juga stabil dan cocok dengan karakteristik data yang digunakan.

## Evaluation
Terdapat dua jenis pengukuran evaluasi metrik yang digunakan, yaitu:
  - MAE (Mean Absolute Error) : Metrik untuk mengukur rata-rata absolut selisih antara nilai hasil prediksi dan nilai aktual pada data. Metrik ini lebih cocok digunakan ketika kita ingin mengukur kesalahan prediksi secara langsung dan linier, tanpa terlalu memprioritaskan kesalahan besar (misalnya, fluktuasi harga saham yang tidak terlalu jauh dari nilai sebenarnya).
  - MSE (Mean Squared Error) : Metrik untuk mengukur rata-rata kuadrat selisih antara nilai yang diprediksi dan nilai aktual. Dalam prediksi harga saham, terutama ketika harga dapat bergerak cukup fluktuatif, MSE dapat memberikan penalti lebih berat pada kesalahan besar, yang membantu model menjadi lebih sensitif terhadap perubahan harga signifikan. Hal ini akan mengurangi risiko model memprediksi harga saham dengan kesalahan besar yang tidak terdeteksi.

Formula untuk MAE yaitu:

Formula untuk MSE yaitu:

Evaluasi metrik pada ketiga model dilakukan setelah proses pelatihan dilakukan. Penjelasan terkait hasil pengujian dengan model ARIMA adalah sebagai berikut:
  - MAE (Mean Absolute Error) sebesar 0.0536 menunjukkan bahwa secara rata-rata, prediksi meleset sekitar 0.0536 unit dari nilai aktual.
  - MSE (Mean Squared Error) sebesar 0.0054 relatif kecil, namun ini disebabkan oleh prediksi yang konsisten di sekitar nilai rata-rata, bukan karena model berhasil menangkap pola volatilitas.
  - Model ARIMA ini cenderung memprediksi nilai mendekati rata-rata (mean reverting) dan tidak menangkap volatilitas harga saham.

Penjelasan terkait hasil pengujian dengan model LSTM yaitu:
  - MAE sebesar 0.5035 menunjukkan rata-rata kesalahan prediksi cukup signifikan.
  - MSE sebesar 0.3469 lebih tinggi dari model ARIMA sebelumnya (karena penggunaan data untuk ARIMA melewati proses difference), mengindikasikan bahwa meskipun LSTM berusaha menangkap volatilitas, akurasi prediksinya masih kurang memuaskan.
  - Model LSTM mampu untuk menangkap beberapa pola penurunan tajam pada data (downspikes), tetapi gagal memprediksi dengan tepat pola kenaikan yang dominan pada data aktual.

Penjelasan terkait hasil pengujian dengan model GRU yaitu:
  - MAE sebesar 0.4531 menunjukkan rata-rata kesalahan prediksi masih cukup signifikan, namun hasilnya lebih baik dari LSTM.
  - MSE sebesar 0.2909 lebih tinggi dari model LSTM, namun tetap akurasi prediksinya masih kurang memuaskan.
  - Model GRU mampu untuk menangkap beberapa pola penurunan tajam pada data (downspikes), tetapi gagal memprediksi dengan tepat pola kenaikan yang dominan pada data aktual.

Kesimpulan akhir dari percobaan ketiga model ini yaitu:
  - Hasil pemodelan yang terbaik dari 3 metode yang dicoba (ARIMA, LSTM, dan GRU) diperoleh dengan menggunakan model GRU.
  - Hasil forecasting menggunakan ARIMA kurang bisa menangkap volatilitas pada data dengan baik
  - Hasil forecasting dengan LSTM & GRU agak sedikit mampu menangkap pola penurunan tajam pada data, namun gagal memprediksi dengan tepat pola kenaikan dominan

## Daftar Pustaka
[[1]]Dey, P., Hossain, E., Hossain, Md. I., Chowdhury, M. A., Alam, Md. S., Hossain, M. S., & Andersson, K. (2021). Comparative Analysis of Recurrent Neural Networks in Stock Price Prediction for Different Frequency Domains. Algorithms, 14(8), 251. https://doi.org/10.3390/a14080251

[[2]]Xiao, R., Feng, Y., Yan, L., & Ma, Y. (2022). Predict stock prices with ARIMA and LSTM (arXiv:2209.02407). arXiv. https://doi.org/10.48550/arXiv.2209.02407

