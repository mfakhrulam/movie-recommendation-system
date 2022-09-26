# Laporan Proyek Machine Learning Sistem Rekomendasi - Muhammad Fakhrul Amin

## Project Overview

Di masa sekarang ini, kuantitas transaksi data yang terjadi setiap menitnya meningkat secara eksponensial. Jumlah data yang berada dalam internet menjadi semakin banyak dan dari sekian banyaknya data tersebut tidak semuanya sesuai dengan yang diinginkan pengguna[[1]](#1). Data yang sangat besar itu jika tidak diproses dengan baik akan terbuang percuma. Dalam beberapa kasus, pengguna perlu melakukan pencarian beberapa kali sebelum menemukan hal yang mereka cari. Untuk itu, diperlukan suatu sistem yang memberikan rekomendasi kepada pengguna terkait dengan informasi yang mereka inginkan berdasarkan informasi relevan yang diambil dari informasi pengguna.  

Sistem rekomendasi dapat menyaring informasi dari internet dan menyarankan informasi yang paling sesuai dengan yang dibutuhkan pengguna sesuai preferensi mereka. Sistem rekomendasi ini secara luas dibagi menjadi tiga jenis, yaitu _content-based filtering_, _collaborative filtering_, dan _hybrid method_[[2]](#2). Dari ketiga jenis tersebut, yang akan dipakai dalam proyek ini adalah _content-based filtering_ dan _collaborative filtering_ untuk proses pemberian saran atau rekomendasi _movie_/film kepada pengguna. Hal ini sangat penting untuk diterapkan agar pengguna merasa nyaman dan senang, sehingga mereka dapat tetap memakai layanan/aplikasi yang dibuat.  

## Business Understanding

Berdasarkan kondisi yang telah diuraikan sebelumnya, akan dikembangkan suatu sistem rekomendasi untuk memberikan rekomendasi atau saran terkait _movie_/film sesuai dengan preferensi atau film yang telah ditonton sebelumnya. Sebelum itu, berikut adalah pernyataan masalah dan tujuan atau _goals_ yang ingin diraih.

### Problem Statements

- Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik _content-based filtering_?
- Dengan data _rating_ yang Anda miliki, bagaimana perusahaan dapat merekomendasikan film lain yang mungkin disukai dan belum pernah ditonton oleh pengguna? 

### Goals

- Menghasilkan sejumlah rekomendasi film yang dipersonalisasi untuk pengguna dengan teknik _content-based filtering_.
- Menghasilkan sejumlah rekomendasi film yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya dengan teknik _collaborative filtering_.

### Solution statements

- Menggunakan pendekatan _content-based filtering_ dengan mencari hubungan antara fitur genre dengan judul film menggunakan _cosine similarity_.  
- Menggunakan pendekatan _collaborative filtering_ dengan mencari hubungan antara _user_, _movie_, dan _rating_ menggunakan model _neural network_.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data film yang dirilis sejak tahun 1996 sampai 2016. Dataset ini terdiri dari satu folder berisi empat file kecil untuk diolah dan satu file berisi metadata dari kumpulan film. Empat file yang ada di dalam folder itu adalah `links.csv`, `movies.csv`, `ratings.csv`, dan `tags.csv`. Namun, dari empat file tersebut, yang digunakan untuk membentuk sistem rekomendasi hanya `movies.csv` dan `ratings.csv`. File `movies.csv` memiliki 9742 baris dan 3 kolom, sementara file `ratings.csv` memiliki 100836 baris dan 4 kolom.  

Adapun sumber asli data ini diambil dari [Movie lens](https://movielens.org), tetapi kemudian diolah dan saya mendapat hasil olahan tersebut dari _kaggle_, sumber data dapat diakses melalui tautan berikut: [Movie Recommendation Data](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data)

### Variabel-variabel pada _Movie Recommendation Data_ dataset adalah sebagai berikut:

- movies: merupakan identitas film/_movie_
  - movieId: kode unik pengidentifikasi film/_movie_
  - title: judul film/_movie_
  - genres: genre film/_movie_
- ratings: merupakan penilaian user terhadap _movie_
  - userId: kode unik untuk identitas user yang melakukan penilaian
  - movieId: kode unik pengidentifikasi film/_movie_ yang dinilai
  - rating: rating atau penilaian film/_movie_
  - timestamp: waktu dilakukannya penilaian

### Exploratory Data Analysis

Hal yang pertama perlu dilakukan setelah memuat dataset yaitu mengetahui isi, info, dan deskripsi dari data. Proses eksplorasi data ini dilakukan pada kedua variabel `movies` dan `ratings`.  

#### Movies

Setelah mengetahui isi dan info dari data `movies`, bisa dilihat bahwa dalam judul (_title_) film terdapat tahun ketika film tersebut rilis. Fitur ini dapat dipisah untuk menampakan persebaran tahun dari film-film tersebut. Selain itu, setelah memisahkan fitur tahun ini, kita dapat melihat bahwa film terlama dirilis pada tahun 1902 dan film terbaru (dalam dataset) dirilis pada tahun 2018, serta didapat pula bahwa ada beberapa film yang tidak diketahui tahun rilisnya. Berikut grafik persebaran tahunnya.   
![Year](https://raw.githubusercontent.com/mfakhrulam/movie-recommendation-system/main/images/01_year.png)    
Gambar 1. Distribusi tahun rilis film  

#### Ratings

Setelah mengetahui isi, info, dan deskripsi dari data `ratings`, didapat beberapa informasi seperti:  
- Tidak semua film mendapat rating oleh pengguna atau _user_. Dilihat dari jumlah film yang diberi rating kurang dari jumlah film yang ada di variabel `movies`.  
- Rating minimum adalah 0.5 dan rating maksimum adalah 5.0  
- _User_ paling banyak memberi rating 4.0 dan paling sedikit memberi rating 0.5. Berikut grafiknya.  
  ![Ratings](https://raw.githubusercontent.com/mfakhrulam/movie-recommendation-system/main/images/02_ratings.png)  
  Gambar 2. Distribusi rating  

## Data Preparation

Data preparation yang dilakukan dalam proyek ini dibagi menjadi dua, yaitu untuk _content-based filtering_ dan _collaborative filtering_.  

### Content-based filtering

- Menggabungkan ratings dengan movies dengan menggunakan perintah `merge` yang ada pada `pandas.Dataframe`. Ini digunakan agar semua dataset digunakan untuk proses pembuatan model.  
- Menghapus _missing value_. Terdapat _missing value_ pada fitur `year` setelah dilakukan pemisahan antara fitur judul dan tahun rilis film. Data yang kosong itu akan dihapus karena tahun rilis film tidak bisa digantikan dengan mean, median, ataupun modus. Untuk menghapusnya gunakan perintah `dropna()` yang ada pada `pandas.Dataframe`.  
- _Drop duplicate_ dan membuat _dataframe_ baru. _Drop duplicate_ yang dimaksud adalah untuk menghilangkan duplikasi pada fitur `title` karena hanya diperlukan data _unique_ untuk proses pemodelan. Untuk menghilangkan duplikasi gunakan perintah `drop_duplicates()` yang ada pada `pandas.Dataframe`. Namun, sebelum proses menghilangkan duplikasi tersebut, lakukan pengurutan terlebih dahulu pada fitur `movieId` menggunakan perintah `sort_values()` dari _library_ `pandas.Dataframe`. Kemudian buat _dataframe_ baru bernama `movies_new` dari data _unique_ tersebut, cukup ambil fitur `movieId`, `title`, `genres`, dan `year` untuk digunakan pada proses pemodelan _content-based filtering_.  

### Collaborative filtering
- Encoding `userId` dan `movieId`. Encoding atau mengkodekan kedua fitur ini bertujuan untuk mempermudah proses _training_, _spliting_, dan pemberian rekomendasi dengan menjadikannya dalam bentuk integer berurutan yang _unique_.  
- Membagi data untuk _training_ dan validasi. Sebelum proses pembagian data, acak dataset dengan menggunakan perintah `sample` dengan parameter `frac=1` dan `random_state=55`. Kemudian data dibagi dengan rasio 80% data latih dan 20% data validasi. Pembagian dataset ini dilakukan untuk menghindari terjadinya _overfitting_ pada data latih ketika diterapkan pada kasus nyata.  

## Modeling

Terdapat dua model berbeda yang disajikan dalam proyek ini, yaitu _content-based filtering_ dan _collaborative filtering_.

### Content-based filtering
Tahap pertama yang dilakukan ketika membuat model ini yaitu menyiapkan mengambil data yang sudah disiapkan pada proses sebelumnya (`movies_new`) dan mengambil sampel data untuk digunakan pada tahap rekomendasi. Kemudian siapkan _TFIDF Vectorizer_ atau `TfidfVectorizer` yang diambil dari _library_ `sklearn.feature_extraction.text` untuk membuat matriks fitur berdasarkan data `genres`.  

Tahap kedua yaitu menghitung _cosine similarity_ yang diambil dari _library_ `sklearn.metrics.pairwise` dengan memasukkan `tfidf_matrix` sebagai argumennya. Kemudian buat _dataframe_ dari dari _cosine similarity_ tersebut dengan data `title` sebagai indeks dan kolomnya.  

Tahap ketiga yaitu membuat fungsi untuk melakukan rekomendasi berdasarkan judul film dan _dataframe_ yang ada di tahap kedua sebagai dasarnya. Ambil _top-n recommendation_ kemudian _drop_ judul film yang diberikan agar tidak muncul dalam rekomendasi. Langkah terakhir yaitu mengambil data sampel yang sudah disiapkan sebelumnya dan memasukkan judul filmnya ke dalam fungsi. Maka rekomendasi judul film yang mirip akan diberikan beserta genrenya.  

_Content-based filtering_ memiliki kelebihan yaitu dapat merekomendasikan film baru tanpa perlu menunggu pengguna lain untuk melakukan rating pada suatu film karena rekomendasi dilakukan berdasarkan konten, dalam hal ini genre, dari film tersebut. Kelemahan dari metode ini adalah terbatasnya rekomendasi hanya pada film-film yang mirip atau berhubungan, sehingga sulit untuk menghasilkan rekomendasi film yang tidak terduga (_serendipitous recommendation_).  

**Hasil rekomendasi dari 5 sampel:**  

Recommendation for:
Double Team => Action  

Tabel 1.1. Rekomendasi _content-based filtering_ untuk judul film "Double Team"  
| |               title  |genres|
|-|----------------------|------|
|0|           Avalanche  |Action|
|1|  Five Deadly Venoms  |Action|
|2|     No Holds Barred  |Action|
|3|     Only the Strong  |Action|
|4|           Fair Game  |Action|

Recommendation for: 
By the Gun => Crime|Drama|Thriller  

Tabel 1.2. Rekomendasi _content-based filtering_ untuk judul film "By the Gun"  
| |                                              title  |                  genres|
|-|-----------------------------------------------------|------------------------|
|0|                                         Mr. Brooks  | Crime\|Drama\|Thriller |
|1|                               Dancer Upstairs, The  | Crime\|Drama\|Thriller |
|2|                                            Villain  | Crime\|Drama\|Thriller |
|3|                                              Fresh  | Crime\|Drama\|Thriller |
|4|  Deadly Outlaw: Rekka (a.k.a. Violent Fire) (Ji...  | Crime\|Drama\|Thriller |

Recommendation for:
Night on Earth => Comedy|Drama  

Tabel 1.3. Rekomendasi _content-based filtering_ untuk judul film "Night on Earth"  
| |             title  |         genres|
|-|--------------------|---------------|
|0|  Boys on the Side  | Comedy\|Drama |
|1|  Last Detail, The  | Comedy\|Drama |
|2|        Paper, The  | Comedy\|Drama |
|3|   Full Monty, The  | Comedy\|Drama |
|4|  Carnal Knowledge  | Comedy\|Drama |

Recommendation for:
Mexican, The => Action|Comedy  

Tabel 1.4. Rekomendasi _content-based filtering_ untuk judul film "Mexican, The"  
| |                title  |          genres|
|-|-----------------------|----------------|
|0|          Tuxedo, The  | Action\|Comedy |
|1|      Game Over, Man!  | Action\|Comedy |
|2|  Beverly Hills Ninja  | Action\|Comedy |
|3|   Disorganized Crime  | Action\|Comedy |
|4|    National Security  | Action\|Comedy |

Recommendation for:
Dangerous Minds => Drama  

Tabel 1.5. Rekomendasi _content-based filtering_ untuk judul film "Dangerous Minds"  
|  |                      title| genres|
|--|---------------------------|-------|
|0 | The Fundamentals of Caring|  Drama|
|1 |               Men of Honor|  Drama|
|2 |                 Tangerines|  Drama|
|3 |                      Mommy|  Drama|
|4 |              Way Back, The|  Drama|


### Collaborative filtering

Tahap pertama yaitu membuat kelas `RecommenderNet` dengan _keras model class_. Model ini menghitung skor kecocokan antara pengguna dan film dengan teknik _embedding_. Setelah proses _embedding_ antara _user_ dan _movie_ selesai, perkalian _dot product_ antara keduanya akan dilakukan. Selain itu, di sini juga ditambahkan bias untuk setiap _user_ dan _movie_. Skor kecocokan ditetapkan dalam skala [0,1] dengan menggunakan fungsi aktivasi sigmoid.  

Tahap kedua yaitu proses inisialisasi, _compile_, dan _train_ model. Lakukan inisialisasi model menggunakan kelas `RecommenderNet` di tahap pertama dengan `num_users`, `num_movie`, dan 50 (embedding size) sebagai parameternya. Lalu _compile_ model dengan loss function menggunakan `BinaryCrossentropy`, optimizer menggunakan `Adam` dengan _learning rate_ 0.001, dan metrik menggunakan `RootMeanSquaredError` atau RMSE. Terakhir lakukan _training_ model menggunakan _dataframe_ yang telah dipisah untuk proses latih dan validasi dengan _epochs_ sebanyak 30 dan _batch size_ sebesar 128.  

Tahap ketiga yaitu mendapatkan rekomendasi film. Untuk mendapatkan rekomendasi film, pertama ambil _user_ acak dan definisikan variabel `movie_not_visited` yang merupakan daftar film yang belum pernah dilihat oleh pengguna dan yang akan dijadikan rekomendasi. Selanjutnya gunakan `model.predict()` untuk mendapatkan rekomendasi filmnya. Maka akan tampil 10 rekomendasi film yang sesuai dengan preferensi pengguna.  

_Collaborative filtering_ memiliki kelebihan yaitu dapat bekerja meskipun konten yang berhubungan dengan item atau user sangat sedikit atau bahkan tidak ada. Sedangkan kelemahannya adalah pendekatan ini sangat memerlukan parameter rating untuk melakukan rekomendasi, sehingga jika ada item yang baru dimasukkan, sistem tidak akan merekomendasikan item tersebut. Masalah ini dinamakan sebagai _cold-start problem_.  

**Hasil rekomendasi untuk user:**  

```
Showing recommendations for users: 413
===========================
Movies with high ratings from user
--------------------------------
Pulp Fiction => Comedy|Crime|Drama|Thriller
Fight Club => Action|Crime|Drama|Thriller
Casino => Crime|Drama
Remember the Titans => Drama
Shanghai Noon => Action|Adventure|Comedy|Western
--------------------------------
Top 10 movie recommendation
--------------------------------
Amadeus => Drama
Grand Day Out with Wallace and Gromit, A => Adventure|Animation|Children|Comedy|Sci-Fi
City of God (Cidade de Deus) => Action|Adventure|Crime|Drama|Thriller
Pianist, The => Drama|War
Boot, Das (Boat, The) => Action|Drama|War
Seventh Seal, The (Sjunde inseglet, Det) => Drama
Touch of Evil => Crime|Film-Noir|Thriller
Glory => Drama|War
Streetcar Named Desire, A => Drama
Manhattan => Comedy|Drama|Romance
```  

Atau jika ditampilkan dalam bentuk tabel, hasil rekomendasinya seperti yang ditunjukkan dalam tabel 2.  

Tabel 2. Rekomendasi _collaborative filtering_ untuk _user_ 413  
|  |                      title| genres|
|--|---------------------------|-------|
|1 | Amadeus|  Adventure\|Animation\|Children\|Comedy\|Sci-Fi |
|2 | Grand Day Out with Wallace and Gromit, A | Adventure\|Animation\|Children\|Comedy\|Sci-Fi |
|3 | City of God (Cidade de Deus) | Action\|Adventure\|Crime\|Drama\|Thriller |
|4 | Pianist, The | Drama\|War |
|5 | Boot, Das (Boat, The) | Action\|Drama\|War |
|6 | Seventh Seal, The (Sjunde inseglet, Det) | Drama |
|7 | Touch of Evil | Crime\|Film-Noir\|Thriller |
|8 | Glory | Drama\|War |
|9 | Streetcar Named Desire, A | Drama |
|10| Manhattan | Comedy\|Drama\|Romance |

## Evaluation

Metriks yang digunakan dalam model _content-based filtering_ adalah _precision_. Metriks ini menghitung jumlah item rekomendasi yang relevan jika dibandingkan dengan semua item yang direkomendasikan. Adapun formulanya dapat ditulis sebagaimana ditunjukkan pada gambar 3.   

![Precision Formula](https://raw.githubusercontent.com/mfakhrulam/movie-recommendation-system/main/images/07_precision_formula.png)    
Gambar 3. Formula _precision_  

Kemudian untuk perhitungannya dilakukan secara manual ditunjukkan pada gambar 4.  

![Precision](https://raw.githubusercontent.com/mfakhrulam/movie-recommendation-system/main/images/08_precision.png)    
Gambar 4. Perhitungan _precision_  

Jika dilihat dari metriks yang ada di gambar 4, dapat disimpulkan bahwa model sudah baik karena nilai precision sudah mencapai nilai 100% atau dalam kata lain sudah sangat presisi.  

Metriks yang digunakan dalam model _collaborative filtering_ adalah _Root Mean Square Error_ (RMSE). Metriks ini dihitung dengan mengkuadratkan _error_ (prediksi – observasi) dibagi dengan jumlah data, lalu diakarkan. Secara umum metriks ini memiliki cara kerja yang sama dengan MAE dan MSE yakni menghtung tingkat _error_ antar nilai prediksi dan nilai sebenarnya. Adapun formulanya dapat ditulis sebagaimana ditunjukkan pada gambar 5.  

![RMSE Formula](https://raw.githubusercontent.com/mfakhrulam/movie-recommendation-system/main/images/06_rmse_formula.png)    
Gambar 5. Formula _Root Mean Square Error_ (RMSE)  

Adapun visualisasi metriksnya dapat dilihat pada gambar 6.  

![RMSE](https://raw.githubusercontent.com/mfakhrulam/movie-recommendation-system/main/images/03_rmse_metric.png)  
Gambar 6. Visualisasi metriks RMSE terhadap model  

Jika dilihat dari metriks yang ada di gambar 6, dapat disimpulkan bahwa model sudah cukup baik karena nilai RMSE sudah cukup rendah yakni sekitar 0.20-0.21.

## Conclusion
Sistem rekomendasi film yang telah dibuat dengan menggunakan pendekatan _content-based filtering_ dan _collaborative filtering_ sudah cukup bagus dan sudah memenuhi _goals_ untuk menghasilkan sejumlah rekomendasi film yang dipersonalisasi untuk pengguna. Model _content-based filtering_ sudah memiliki presisi yang sangat bagus, sementara model _collaborative filtering_ sudah memiliki RMSE yang cukup kecil, sehingga dapat dikatakan bahwa kedua model sudah mencapai _goals_ yang diinginkan. Namun, tidak menutup kemungkinan untuk kedua model ini memperoleh _improvement_ untuk menghasilkan rekomendasi yang lebih baik.  

## References

<a id="1">[1]</a>  [Z. Wang, X. Yu, N. Feng, and Z. Wang, “An improved collaborative movie recommendation system using computational intelligence,” J. Vis. Lang. Comput., vol. 25, no. 6, pp. 667–675, 2014, doi: 10.1016/j.jvlc.2014.09.011.](https://www.sciencedirect.com/science/article/abs/pii/S1045926X14000901)  

<a id="2">[2]</a>  [S. Reddy, S. Nalluri, S. Kunisetti, S. Ashok, and B. Venkatesh, Content-based movie recommendation system using genre correlation, vol. 105. Springer Singapore, 2019. doi: 10.1007/978-981-13-1927-3_42.](https://link.springer.com/chapter/10.1007/978-981-13-1927-3_42)  
