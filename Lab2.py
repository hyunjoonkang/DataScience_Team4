import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Load the dataset
data = pd.read_csv('./bmi_data_lab2.csv')

# Print dataset statiscal data, feature names & data types
print(data.describe())
print()
print(data.info())

# Create subplots for Height and Weight histograms
hei_wei_fig, hei_wei_axes = plt.subplots(2, 5, figsize=(16, 8))
hei_wei_fig.suptitle('Height Weight Histograms')

# Draw histogram for each BMI level
for i in [0.0, 1.0, 2.0, 3.0, 4.0]:
    BMI_group_data = data[data['BMI'] == i]

    # Plot Height histogram
    hei_wei_axes[0, int(i)].hist(BMI_group_data['Height (Inches)'], bins=10)
    hei_wei_axes[0, int(i)].set_title(f'Height (BMI={int(i)})')

    # Plot Weight histogram
    hei_wei_axes[1, int(i)].hist(BMI_group_data['Weight (Pounds)'], bins=10)
    hei_wei_axes[1, int(i)].set_title(f'Weight (BMI={int(i)})')

# Concatenate Height and Weight data for scaling
hei_wei_data = pd.concat(
    [data['Height (Inches)'], data['Weight (Pounds)']], axis=1)

# Apply different scaling methods 1: StandardScaler
S_Scaler = StandardScaler()
S_Scaled = pd.DataFrame(S_Scaler.fit_transform(
    hei_wei_data), columns=hei_wei_data.columns)

# Apply different scaling methods 2: MinMaxScaler
MM_Scaler = MinMaxScaler()
MM_Scaled = pd.DataFrame(MM_Scaler.fit_transform(
    hei_wei_data), columns=hei_wei_data.columns)

# Apply different scaling methods 3: RobustScaler
R_Scaler = RobustScaler()
R_Scaled = pd.DataFrame(R_Scaler.fit_transform(
    hei_wei_data), columns=hei_wei_data.columns)

# Create subplots for scaling comparison
scaleFig, scaleAxes = plt.subplots(1, 4, figsize=(12, 4))

# Plot original data
scaleAxes[0].set_title('Before Scaling')
sns.kdeplot(hei_wei_data, ax=scaleAxes[0])

# Plot data after Standard Scaling
scaleAxes[1].set_title('Standard Scaling')
sns.kdeplot(S_Scaled, ax=scaleAxes[1])

# Plot data after MinMax Scaling
scaleAxes[2].set_title('MinMax Scaling')
sns.kdeplot(MM_Scaled, ax=scaleAxes[2])

# Plot data after Robust Scaling
scaleAxes[3].set_title('Robust Scaling')
sns.kdeplot(R_Scaled, ax=scaleAxes[3])

plt.show()
