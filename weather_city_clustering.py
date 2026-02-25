# %%
import pandas as pd
import io

csv = "Data/weather_db.historical_exports_2:23:2026.csv"
master_df = pd.read_csv(csv)

# %%
df = {} # cities: pd.DataFrame(data)
for _, row in master_df.iterrows():
    city = row['city']
    csv_text = row['csv_content']
    csv_text = csv_text.replace("\\n", "\n")
    df[city] = pd.read_csv(io.StringIO(csv_text))

# Step 1: Decide what to cluster by
    # mean temperature
    # temperature variance
    # seasonal strength
    # daily/annual amplitude
    # trend (warming slope)
    # extreme frequency
    # autocorrelation

# Step 2: Clean and prepare each city
for city, content in df.items():
    content['timestamp'] = pd.to_datetime(content['timestamp'])
    content.sort_values('timestamp', inplace=True)
    content.set_index('timestamp', inplace=True)

# %%
# Step 3: Create features per city
features = []
for city, content in df.items():
    temp = content['tempF'].dropna()

    city_features = {
        "city": city,
        "mean_temp": temp.mean(),
        "min_temp": temp.min(),
        "max_temp": temp.max(),
        "std_temp": temp.std(),
        "range_temp": temp.max() - temp.min(),
        "median_temp": temp.median(),
        "skew_temp": temp.skew(),
        "kurtosis_temp": temp.kurtosis()
    }

    features.append(city_features)

features_df = pd.DataFrame(features)
features_df.set_index("city", inplace=True)

# Step 4: Scale the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)

# %%
# Step 5: Perform clustering
from sklearn.cluster import KMeans

# option A: k-means
km = KMeans(n_clusters=3)
labels = km.fit_predict(X_scaled)
features_df["k-means-label"] = labels

# %%
# option B: cluster by seasonal pattern
import numpy as np
monthly_profiles = []

for city, content in df.items():
    monthly = content['tempC'].resample('M').mean()
    
    # average temperature per month across years
    monthly_avg = monthly.groupby(monthly.index.month).mean()
    
    monthly_profiles.append(monthly_avg.values)

monthly_matrix = np.array(monthly_profiles)

X_scaled = StandardScaler().fit_transform(monthly_matrix)
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X_scaled)
features_df["seasonal-label"] = labels

# %%
# option C: time-series distance
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
time_series_list = []

for city, content in df.items():
    # resample to daily average
    daily = content['tempC'].resample('D').mean().dropna()
    
    time_series_list.append(daily.values)

min_length = min(len(ts) for ts in time_series_list)

time_series_list = [ts[:min_length] for ts in time_series_list]

time_series_array = np.array(time_series_list)
time_series_array = time_series_array.reshape(time_series_array.shape[0], time_series_array.shape[1], 1)

scaler = TimeSeriesScalerMeanVariance()
time_series_array = scaler.fit_transform(time_series_array)

model = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=42)
labels = model.fit_predict(time_series_array)
features_df["time-series-distance-label"] = labels
# %%
features_df
