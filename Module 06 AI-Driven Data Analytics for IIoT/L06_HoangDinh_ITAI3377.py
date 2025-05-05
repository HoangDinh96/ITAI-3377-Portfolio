
import pandas as pd
import numpy as np
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from window_ops.rolling import rolling_mean
import matplotlib.pyplot as plt


# 1. DATA PREPARATION 

df = pd.read_csv('IOT-temp.csv')

df_model = df[['room_id/id', 'noted_date', 'temp']].copy()
df_model.rename(columns={'room_id/id': 'unique_id', 'noted_date': 'ds', 'temp': 'y'}, inplace=True)
df_model['ds'] = pd.to_datetime(df_model['ds'], dayfirst=True)
df_model = df_model.sort_values(['unique_id', 'ds'])

# Missing values
print("🔍 Missing values:\n", df_model.isnull().sum())

# Outliers
z_scores = np.abs((df_model['y'] - df_model['y'].mean()) / df_model['y'].std())
df_model['is_outlier'] = z_scores > 3
print(f"\n⚠️ Outliers detected: {df_model['is_outlier'].sum()}")

# Normalize
print("\n📈 Value range before normalization:")
print(df_model['y'].describe())
scaler = MinMaxScaler()
df_model['y'] = scaler.fit_transform(df_model[['y']])
df_model.drop(columns=['is_outlier'], inplace=True)


# 2. MODEL SELECTION & TRAINING

fcst = MLForecast(
    models=[RandomForestRegressor()],
    lags=[1, 2, 3, 6, 12],
    lag_transforms={1: [(rolling_mean, 3)]},
    date_features=['hour', 'dayofweek'],
    freq='h'
)

fcst.fit(
    df=df_model,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
    static_features=[]
)

forecast = fcst.predict(12)
forecast.rename(columns={forecast.columns[-1]: 'y_pred'}, inplace=True)
print("\n🧠 Forecast head:\n", forecast.head())


# 4. EVALUATION

room_id = df_model['unique_id'].iloc[0]
recent = df_model[df_model['unique_id'] == room_id].tail(12)
preds = forecast[forecast['unique_id'] == room_id]

y_true = recent['y'].values
y_pred = preds['y_pred'].values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mase = mae / mean_absolute_error(df_model['y'][1:], df_model['y'][:-1])

print("\n📊 Holdout Evaluation:")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"MASE: {mase:.4f}")

# ROLLING-ORIGIN CROSS VALIDATION

print("\n🔁 Rolling-Origin Cross-Validation (Manual)")

window_size = 1
horizon = 12
metrics = []

df_model = df_model.sort_values(['unique_id', 'ds'])

for i in range(window_size):
    cutoff_index = -((window_size - i) * horizon)

    if abs(cutoff_index) >= len(df_model):
        continue

    train = df_model.iloc[:cutoff_index].copy()
    test = df_model.iloc[cutoff_index:cutoff_index + horizon].copy()

    fcst.fit(
        df=train,
        id_col='unique_id',
        time_col='ds',
        target_col='y',
        static_features=[]
    )

    preds = fcst.predict(horizon)
    preds.rename(columns={preds.columns[-1]: 'y_pred'}, inplace=True)

    merged = pd.merge(test, preds, on=['unique_id', 'ds'], how='inner')
    if merged.empty:
        continue

    mae = mean_absolute_error(merged['y'], merged['y_pred'])
    mse = mean_squared_error(merged['y'], merged['y_pred'])
    metrics.append({'cutoff': test['ds'].min(), 'MAE': mae, 'MSE': mse})

if metrics:
    cv_df = pd.DataFrame(metrics)
    print("\n📉 Rolling-Origin CV Results:\n", cv_df)
else:
    print("\n⚠️ Not enough data to perform rolling-origin cross-validation.")


# OPTIONAL: PLOT FORECAST

#plt.figure(figsize=(10, 5))
#plt.plot(recent['ds'], y_true, label='Actual')
#plt.plot(preds['ds'], y_pred, label='Forecast', linestyle='--')
#plt.title(f"Forecast for Room: {room_id}")
#plt.xlabel('Time')
#plt.ylabel('Normalized Temperature')
#plt.legend()
#plt.tight_layout()
#plt.show()

# 5. GENERATIVE MODELING (VAE)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("\n🚀 Starting VAE-based synthetic data generation...")

# PARAMETERS
sequence_length = 24  # 1 day of hourly data
latent_dim = 16
num_sequences = 500  # how many synthetic sequences to generate

# CREATE SEQUENCES ACROSS ALL ROOMS
def create_sequences(df, seq_len):
    sequences = []
    for _, group in df.groupby('unique_id'):
        y = group['y'].values
        for i in range(len(y) - seq_len):
            seq = y[i:i+seq_len]
            sequences.append(seq)
    return np.array(sequences)

real_sequences = create_sequences(df_model, sequence_length)

# DEFINE VAE MODEL
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def build_vae(input_dim, latent_dim):
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(encoder_inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation="relu")(latent_inputs)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # VAE model
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="loss")

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(keras.losses.mse(data, reconstruction))
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            return {"loss": self.total_loss_tracker.result()}


    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    return vae, encoder, decoder

vae, encoder, decoder = build_vae(sequence_length, latent_dim)

# TRAIN THE VAE
vae.fit(real_sequences, epochs=30, batch_size=128, verbose=1)

# GENERATE SYNTHETIC SEQUENCES 
z_samples = np.random.normal(size=(num_sequences, latent_dim))
synthetic_sequences = decoder.predict(z_samples)

# CONVERT SYNTHETIC SEQUENCES TO DataFrame FORMAT 
synthetic_df = []
base_time = pd.to_datetime(df_model['ds'].max()) + pd.Timedelta(hours=1)
for i, seq in enumerate(synthetic_sequences):
    for j, val in enumerate(seq):
        synthetic_df.append({
            'unique_id': f'synthetic_{i}',
            'ds': base_time + pd.Timedelta(hours=j),
            'y': val
        })
synthetic_df = pd.DataFrame(synthetic_df)

# AUGMENT ORIGINAL DATA 
augmented_df = pd.concat([df_model[['unique_id', 'ds', 'y']], synthetic_df], ignore_index=True)
augmented_df = augmented_df.sort_values(['unique_id', 'ds'])

print("\n🧪 Augmented dataset created. Retraining model...")

# RETRAIN FORECASTING MODEL WITH AUGMENTED DATA 
fcst.fit(
    df=augmented_df,
    id_col='unique_id',
    time_col='ds',
    target_col='y',
    static_features=[]
)

forecast_aug = fcst.predict(12)
forecast_aug.rename(columns={forecast_aug.columns[-1]: 'y_pred'}, inplace=True)

# EVALUATE AGAIN 
preds_aug = forecast_aug[forecast_aug['unique_id'] == room_id]
mae_aug = mean_absolute_error(y_true, preds_aug['y_pred'].values)
mse_aug = mean_squared_error(y_true, preds_aug['y_pred'].values)
mase_aug = mae_aug / mean_absolute_error(df_model['y'][1:], df_model['y'][:-1])

print("\n📊 Evaluation After Augmentation:")
print(f"MAE:  {mae_aug:.4f} (before: {mae:.4f})")
print(f"MSE:  {mse_aug:.4f} (before: {mse:.4f})")
print(f"MASE: {mase_aug:.4f} (before: {mase:.4f})")
