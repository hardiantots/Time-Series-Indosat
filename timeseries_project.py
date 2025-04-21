# %% [markdown]
# ## Import Library & Data Loading

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import yfinance as yf
import itertools
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# %%
data = yf.download('ISAT.JK', start='2018-06-01', end='2025-04-01')
data = data.reset_index()
data.to_csv('indosat_stock_data.csv', index=False)
stock_data = pd.read_csv('indosat_stock_data.csv')
stock_data = stock_data.iloc[1:]

# %% [markdown]
# ## Data Understanding

# %% [markdown]
# Tahapan ini berfokus pada mengetahui unsur-unsur dari data yang telah di-load. Beberapa diantaranya yaitu jumlah baris-kolom, tipe data setiap kolom, mengecek statistik deskriptif, identifikasi missing value atau outlier, visualisasi distribusi dan pola pada data, dan lain sebagainya. 

# %% [markdown]
# ##### Mencari info tipe data setiap kolom & jumlah baris-kolom

# %%
stock_data.head()

# %%
stock_data.info()

# %%
print(f"Jumlah data (baris, kolom): {stock_data.shape}")

# %% [markdown]
# Hasilnya menyatakan bahwa:
# - Dataset ini terdiri dari 1679 data dan 6 kolom (Date, Close, High, Low, Open, Volume)
# - Semua tipe datanya masih belum sesuai (kolom angka harusnya berbentuk float dan integer)

# %%
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

list_cols_to_float = ['Close', 'Open', 'High', 'Low']
stock_data[list_cols_to_float] = stock_data[list_cols_to_float].astype('float64').round(5)

stock_data['Volume'] = stock_data['Volume'].astype('int64')

# %%
stock_data.info()

# %% [markdown]
# Tindakan pengubahan tipe data harus dilakukan lebih awal untuk memudahkan dalam analisis-analisis berikutnya

# %% [markdown]
# ##### Mengecek Statistik Deskriptif Kolom

# %%
stock_data.describe()

# %% [markdown]
# ##### Identifikasi missing value, data duplikat, dan outlier

# %%
print("Jumlah Missing Value pada setiap kolom: ")
stock_data.isna().sum()

# %%
print(f"Jumlah data yang terduplikat: {stock_data.duplicated().sum()}")

# %%
cols = ['Close', 'High', 'Low', 'Open', 'Volume']
fig, axes = plt.subplots(len(cols), 1, figsize=(15, 10))

for ax, col in zip(axes, cols):
    sns.boxplot(x=col, data=stock_data, ax=ax)
    ax.set_title(f'Plot Boxplot {col}')

plt.tight_layout()
plt.show()

# %% [markdown]
# - Tidak terdapat missing value maupun data yang terduplikat
# - Terdapat outlier pada kolom volume (untuk saat ini belum akan ada tindakan lanjutan mengingat data yang terbatas)

# %% [markdown]
# ##### Visualisasi Distribusi & Pola pada Data

# %%
plt.figure(figsize=(30, 7))

for i, col in enumerate(cols):
    plt.subplot(1, len(cols), i + 1)
    sns.histplot(stock_data[col], bins=15, kde=True)
    plt.title(f'Plot Histogram {col}')
    plt.ylabel(col)

# %%
plt.figure(figsize=(16, 24))

combinations = list(itertools.combinations(cols, 2))

for i, (x, y) in enumerate(combinations, 1):
    plt.subplot(len(combinations)//2 + 1, 2, i)
    sns.scatterplot(data=stock_data, x=x, y=y, alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y}')

plt.tight_layout()
plt.suptitle("Scatter Plot", y=1.02)
plt.show()

# %% [markdown]
# Hasil analisa distribusi data:
# - Distribusi pada kolom fitur Close, Open, High, Low memiliki banyak puncak dan tidak simetris. Distribusi terlihat agak terpisah dalam beberapa interval. Ini bisa mengindikasikan adanya beberapa kelompok atau segmentasi dalam data harga saham tersebut.
# - Distribusi volume sangat right-skewed (miring ke kanan), dengan banyak nilai kecil dan beberapa nilai yang sangat besar (outlier).
# - Transformasi akan dilakukan pada tahapan berikutnya dalam upaya mengurangi skewness.
# 
# Hasil analisa hubungan pola data:
# - Terdapat beberapa pasangan kolom data yang memiliki korelasi linear dan positif, yaitu:
#     - Close-High | Close-Low | Close-Open
#     - High-Low | High-Open
#     - Low-Open
# - Ini menandakan bahwa 4 kolom (Close, High, Low, Open) saling berkorelasi kuat secara linear
# - Untuk hubungan 4 kolom tadi terhadap volume, hasilnya menyatakan bahwa tidak memiliki hubungan linear yang kuat

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# Berdasarkan hasil data understanding di bagian sebelumnya, akan dilakukan analisis-analisis untuk mengetahui informasi yang bisa diperoleh dari dataset ini

# %% [markdown]
# ##### Melihat tren data dari waktu ke waktu

# %%
def graph_feature(data):
    pairs = [('Close', 'Open'), ('High', 'Low')]
    for pair in pairs:  
        plt.figure(figsize=(20, 6))  
 
        plt.plot(data['Date'], data[pair[0]], label=pair[0])  
        plt.plot(data['Date'], data[pair[1]], label=pair[1])  

        plt.xlabel('Time Period')
        plt.ylabel('Value')  
        plt.title(f'Time Series Graph {pair[0]} & {pair[1]}')  
        plt.legend()  
        plt.grid(True)  
        plt.tight_layout()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        plt.xticks(rotation=45)
        plt.show()

# %%
graph_feature(stock_data)

# %% [markdown]
# Hasil analisa untuk grafik waktu pada data:
# - Nilai saham cenderung mengalami peningkatan harga hingga mengalami puncaknya pada pertengahan tahun 2024.
# - Setelah mengalami puncak harga saham, perlahan-lahan nilainya kembali mengalami penurunan hingga tahun 2025 

# %% [markdown]
# ##### Megecek beberapa hal pada setiap kolom dengan seasonal compose

# %%
n_cols = len(cols)  

for i, col in enumerate(cols):
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(16, 14))

    data = stock_data[col].dropna()
    result = seasonal_decompose(data, model='additive', period=24) 

    ax[0].plot(result.observed)  
    ax[0].set_title(f'{col} - Observed')  
    
    ax[1].plot(result.trend)  
    ax[1].set_title(f'{col} - Trend')
    
    ax[2].plot(result.seasonal)  
    ax[2].set_title(f'{col} - Seasonal')  
    
    ax[3].plot(result.resid)  
    ax[3].set_title(f'{col} - Residual')  

plt.tight_layout()  
plt.show() 

# %% [markdown]
# Hasil analisa untuk grafik Open, Close, High, Low:
# - Observed : Menunjukkan peningkatan harga. Namun pada bagian akhir dari data, nampak mengalami penurunan   
# - Trend : Cenderung mengalami kenaikan bertahap, lalu sedikit mengalami penurunan, kemudian akan kembail mengalami kenaikan
# - Seasonal : Polanya menunjukkan siklus yang teratur.
# - Residual : Dalam grafik ini, terdapat nilai dengan outlier.
# 
# Hasil analisa untuk grafik Volume:
# - Observed : Menunjukkan tren stabil, dengan adanya beberapa waktu tertentu yang mengalami kenaikan volume.
# - Trend : Cenderung mengalami tren konstan meskipun terdapat beberapa lonjakan pada periode waktu tertentu.
# - Seasonal : Polanya menunjukkan siklus yang teratur.
# - Residual : Terdapat beberapa nilai yang mengandung outlier.

# %% [markdown]
# ##### Melihat grafik ACF dan PACF di setiap kolomnya

# %%
fig, axes = plt.subplots(len(cols), 2, figsize=(12, 8))

for i, col in enumerate(cols):  
    data = stock_data[col].dropna()
    plot_acf(data, lags=50, ax=axes[i, 0], title=f'ACF of {col}')  
    plot_pacf(data, lags=50, ax=axes[i, 1], title=f'PACF of {col}')  

plt.tight_layout()  
plt.show()

# %% [markdown]
# Hasil grafik ACF dan PACF untuk variabel:
# 
# - Close, High, Low, Open: ACF semuanya menunjukkan pola yang sangat lambat menurun (slow decay) dengan nilai tinggi dan mendekati 1 hingga lag yang jauh. Ini mengindikasikan bahwa data series ini bersifat non-stasioner, atau memiliki tren jangka panjang. Nilai saat ini sangat berkorelasi dengan nilai di masa lalu dalam jangka waktu yang panjang. PACF menunjukkan spike signifikan hanya pada lag 1 & lag 2, kemudian langsung turun mendekati nol untuk lag berikutnya. Ini biasanya mengindikasikan pola AR(1), yaitu data dipengaruhi dominan oleh nilai sebelumnya satu periode.
# - Volume: ACF Volume menurun secara bertahap namun lebih cepat dibandingkan dengan harga (Close, High, Low, Open), menunjukkan data ini memiliki ketergantungan waktu yang lebih lemah dan kemungkinan lebih mendekati stasioner. PACF Volume menunjukkan nilai terbesar di lag 1 dan beberapa lag awal signifikan, kemudian cukup cepat menurun mendekati nol. Hal ini mengindikasikan pola AR atau ARMA dengan orde rendah. Volume bisa jadi lebih mudah distasionerkan

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# Beberapa hal akan dilakukan pada bagian ini, seperti:
# - Standarisasi data
# - Set kolom 'Date' sebagai index
# - Proses train-test split

# %%
scaler = StandardScaler()
stock_data[cols] = scaler.fit_transform(stock_data[cols])
stock_data.describe().round(1)

# %% [markdown]
# Metode Standarisasi digunakan untuk mengubah skala data dikarenakan bentuk distribusinya tidak ada yang normal/bell-curved

# %%
stock_data.set_index('Date', inplace=True)

# %% [markdown]
# ##### Data Preparation untuk pemodelan dengan menggunakan ARIMA

# %%
def check_stationarity(series, name=''):
    result = adfuller(series.dropna())
    print(f'ADF Statistic {name}: {result[0]}')
    print(f'p-value: {result[1]}')

for col in cols:
    check_stationarity(stock_data[col], name=col)

# %% [markdown]
# Hasil dengan ADF test menyatakan bahwa semua kolom data perlu didifferencing karena nilai p-value > 0.05. Ini dilakukan untuk menjadikan data stasioner (karakteristik statistik tidak berubah sepanjang waktu) agar bisa pemodelan dengan pendekatan time-series (ARIMA)

# %%
diff_data = stock_data.diff().dropna()

# %% [markdown]
# Splliting data untuk pemodelan dengan ARIMA

# %%
split_index = int(len(diff_data) * 0.7)
train = diff_data.iloc[:split_index]
valid = diff_data.iloc[split_index:]

# %%
print(f"Jumlah data train: {train.shape}")
print(f"Jumlah data valid: {valid.shape}")

# %% [markdown]
# #### Data Preparation untuk pemodelan dengan Deep Learning

# %%
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)

# %%
prices = stock_data['Close'].values

# %%
SPLIT_TIME = int(len(prices) * 0.7)
train_price = prices[:SPLIT_TIME]
valid_price = prices[SPLIT_TIME:]

# %%
print(f"Jumlah data train: {train_price.shape}")
print(f"Jumlah data valid: {valid_price.shape}")

# %%
BATCH_SIZE = 4
WINDOW_SIZE = 2
SHUFFLE_BUFFER = 1000

train_set = windowed_dataset(series=train_price, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER)
valid_set = windowed_dataset(series=valid_price, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER)

# %% [markdown]
# ## Modelling Process

# %% [markdown]
# Dalam pemodelan ini saya membandingkan 3 jenis model, yaitu ARIMA, LSTM, dan GRU

# %% [markdown]
# #### ARIMA

# %%
model = ARIMA(train['Close'], order=(1,1,2))
result = model.fit()
print('Fitted ARIMA model for Close Prices')
print(result.summary())

# %%
forecast_horizon = len(valid)

forecast = result.get_forecast(steps=forecast_horizon)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

forecast_df = pd.DataFrame({'forecast': forecast_mean})

# %% [markdown]
# #### LSTM

# %%
tf.random.set_seed(42)

early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath='model_best.h5', monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

callbacks = [early_stop_callback, checkpoint, reduce_lr_callback]

# %%
model_lstm = Sequential([
    LSTM(128, input_shape=(WINDOW_SIZE, 1),return_sequences=True, activation='tanh'),
    LSTM(256, return_sequences=True, activation='tanh'),
    BatchNormalization(),
    LSTM(128, return_sequences=True, activation='tanh'),
    LSTM(64, return_sequences=True, activation='tanh'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model_lstm.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

# %%
hist_lstm = model_lstm.fit(train_set, validation_data=valid_set, epochs=50, callbacks=callbacks)

# %% [markdown]
# #### GRU

# %%
model_gru = Sequential([
    GRU(128, input_shape=(WINDOW_SIZE, 1),return_sequences=True, activation='tanh'),
    GRU(256, return_sequences=True, activation='tanh'),
    BatchNormalization(),
    GRU(128, return_sequences=True, activation='tanh'),
    GRU(64, return_sequences=True, activation='tanh'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model_gru.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

# %%
hist_gru = model_gru.fit(train_set, validation_data=valid_set, epochs=50, callbacks=callbacks)

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ##### Evaluasi pada hasil Model ARIMA

# %%
mse = mean_squared_error(valid['Close'], forecast_mean)
mae = mean_absolute_error(valid['Close'], forecast_mean)
print(f'MAE for Close Price: {mae:.4f}')
print(f'MSE for Close Price: {mse:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(valid.index, valid['Close'], label='Actual', color='blue')
plt.plot(valid.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(valid.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Close Price Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Close Price (Differenced)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Pemaparan terkait Evaluasi Metrik dari hasil pemodelan dengan ARIMA

# %% [markdown]
# - MAE (Mean Absolute Error) sebesar 0.0527 menunjukkan bahwa secara rata-rata, prediksi meleset sekitar 0.0536 unit dari nilai aktual.
# - MSE (Mean Squared Error) sebesar 0.0053 relatif kecil, namun ini disebabkan oleh prediksi yang konsisten di sekitar nilai rata-rata, bukan karena model berhasil menangkap pola volatilitas.
# - Model ARIMA ini cenderung memprediksi nilai mendekati rata-rata (mean reverting) dan tidak menangkap volatilitas harga saham.

# %% [markdown]
# ##### Evaluasi pada Model LSTM

# %%
def plot_metrics(history_model):
    available_keys = history_model.history.keys()
    print("Available metrics:", available_keys)

    loss = history_model.history.get('loss')
    val_loss = history_model.history.get('val_loss')
    mae = history_model.history.get('mae')
    val_mae = history_model.history.get('val_mae')
    mse = history_model.history.get('mse')
    val_mse = history_model.history.get('val_mse')

    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()

    if mae and val_mae:
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, mae, label='Train MAE')
        plt.plot(epochs_range, val_mae, label='Val MAE')
        plt.title('Mean Absolute Error')
        plt.legend()

    if mse and val_mse:
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, mse, label='Train MSE')
        plt.plot(epochs_range, val_mse, label='Val MSE')
        plt.title('Mean Squared Error')
        plt.legend()

    plt.tight_layout()
    plt.show()

# %%
plot_metrics(hist_lstm)

# %% [markdown]
# Hasil dari pemodelan menggunakan LSTM menunjukkan progress yang baik pada train set, namun pada validation set performanya masih cenderung kurang baik.

# %%
X_valid = []
y_valid = []

for x_batch, y_batch in valid_set:
    X_valid.append(x_batch.numpy())
    y_valid.append(y_batch.numpy())

X_valid = np.concatenate(X_valid, axis=0)
y_valid = np.concatenate(y_valid, axis=0)

# %%
def plot_forecasting(model, model_name):
    y_pred = model.predict(X_valid)
    
    if len(y_pred.shape) == 3:
        y_pred = y_pred[:, -1, 0]
    if len(y_valid.shape) == 3:
        y_valid_plot = y_valid[:, -1, 0]
    else:
        y_valid_plot = y_valid
    
    mse = mean_squared_error(y_valid_plot, y_pred)
    mae = mean_absolute_error(y_valid_plot, y_pred)
    print(f'MAE for {model_name}: {mae:.4f}')
    print(f'MSE for {model_name}: {mse:.4f}')

    plt.figure(figsize=(12, 6))
    plt.plot(y_valid_plot, label='Actual', color='blue')
    plt.plot(y_pred, label='Forecast', color='red')
    plt.title(f'{model_name} Forecast vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
plot_forecasting(model_lstm, "LSTM")

# %% [markdown]
# Pemaparan terkait Evaluasi Metrik dari menggunakan model LSTM

# %% [markdown]
# - MAE sebesar 0.5035 menunjukkan rata-rata kesalahan prediksi cukup signifikan.
# - MSE sebesar 0.3469 lebih tinggi dari model ARIMA sebelumnya (karena penggunaan data untuk ARIMA melewati proses difference), mengindikasikan bahwa meskipun LSTM berusaha menangkap volatilitas, akurasi prediksinya masih kurang memuaskan.
# - Model LSTM mampu untuk menangkap beberapa pola penurunan tajam pada data (downspikes), tetapi gagal memprediksi dengan tepat pola kenaikan yang dominan pada data aktual.

# %% [markdown]
# #### Evaluasi pada Model GRU

# %%
plot_metrics(hist_gru)

# %% [markdown]
# Hasil dari pemodelan menggunakan GRU menunjukkan progress yang baik pada train set, namun pada validation set performanya masih cenderung kurang baik meskipun hasilnya sedikit lebih baik dari LSTM.

# %%
plot_forecasting(model_gru, "GRU")

# %% [markdown]
# Pemaparan terkait Evaluasi Metrik dari menggunakan model GRU

# %% [markdown]
# - MAE sebesar 0.4531 menunjukkan rata-rata kesalahan prediksi masih cukup signifikan, namun hasilnya lebih baik dari LSTM.
# - MSE sebesar 0.2909 lebih tinggi dari model LSTM, namun tetap akurasi prediksinya masih kurang memuaskan.
# - Model GRU mampu untuk menangkap beberapa pola penurunan tajam pada data (downspikes), tetapi gagal memprediksi dengan tepat pola kenaikan yang dominan pada data aktual.

# %% [markdown]
# #### Kesimpulan Akhir dari Hasil Pemodelan

# %% [markdown]
# - Hasil pemodelan yang terbaik dari 3 metode yang dicoba (ARIMA, LSTM, dan GRU) diperoleh dengan menggunakan model GRU.
# - Hasil forecasting menggunakan ARIMA kurang bisa menangkap volatilitas pada data dengan baik
# - Hasil forecasting dengan LSTM & GRU agak sedikit mampu menangkap pola penurunan tajam pada data, namun gagal memprediksi dengan tepat pola kenaikan dominan 
