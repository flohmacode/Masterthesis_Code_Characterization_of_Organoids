import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

'''This Script compares posterior distributions of two Experiments'''

def load_to_dataframe(npy_path):
    """Loads the .npy file containing an array of dictionaries and converts it to a pandas DataFrame."""
    raw_data = np.load(npy_path, allow_pickle=True)

    # Handle case where numpy array is wrapped or unextracted
    if raw_data.ndim == 0:
        raw_data = raw_data.item()

    # Convert the array of dictionaries into a clean DataFrame
    df = pd.DataFrame(list(raw_data))

    # Clean up numpy types inside the cells (e.g., converting np.float64 to normal floats)
    return df.apply(pd.to_numeric, errors="ignore")


# 1. Define path setups

jojo_path = f"./modeling/parameters/jojo_april_linear_accepted_parameter_numbers.npy"
leupold_path = f"./modeling/parameters/leupold_feb_linear_accepted_parameter_numbers.npy"

# 2. Load data into DataFrames
df_jojo = load_to_dataframe(jojo_path)
df_leupold = load_to_dataframe(leupold_path)

# Add a label column to distinguish them when merged
df_jojo["Dataset"] = "3. Experiment"
df_leupold["Dataset"] = "2. Experiment"

# Combine the datasets for easier plotting with seaborn
df_combined = pd.concat([df_jojo, df_leupold], ignore_index=True)

# 3. Separate parameters that vary from those that are fixed/constant
all_numerical_cols = df_jojo.select_dtypes(include=[np.number]).columns
dynamic_params = [
    col for col in all_numerical_cols if df_combined[col].nunique() > 1
]
constant_params = [
    col for col in all_numerical_cols if df_combined[col].nunique() == 1
]

print(f"Plotting dynamic parameters: {dynamic_params}")
print(f"Skipping constant parameters: {constant_params}")

# 4. Dynamically generate grid plots for changing parameters
num_params = len(dynamic_params)
cols = 3  # Number of columns in our grid
rows = (num_params + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
axes = axes.flatten()  # Flatten to iterate easily

my_colors = {
    "3. Experiment": "#0C68C4",   # Deep Navy
    "2. Experiment": "#F8720C"   # Burnt Orange
}

for i, param in enumerate(dynamic_params):
    ax = axes[i]

    # Plot overlapping distribution curves
    sns.kdeplot(
        data=df_combined,
        x=param,
        hue="Dataset",
        fill=True,
        common_norm=False,
        palette=my_colors,
        alpha=0.4,
        linewidth=2,
        ax=ax,
    )

     
    ax.set_title(f"Distribution of {param}", fontsize=12)
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.6)

# Hide any leftover empty subplots in the grid
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()