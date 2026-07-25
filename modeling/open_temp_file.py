import matplotlib.pyplot as plt
import numpy as np

timestamps = []
values = []

with open('./modeling/processed_data/temperature_file_jojo0409/260218_4.tem') as file:
    for line in file:
        # line.split() handles tabs and spaces automatically and drops trailing newlines
        parts = line.split() 
        
        try:
            # Dynamically handle lines with or without the date/time prefix
            if len(parts) == 7:
                ts = float(parts[2])   # High-resolution timestamp column
                val = float(parts[4])  # Temperature value column
            elif len(parts) == 5:
                ts = float(parts[0])   # High-resolution timestamp column
                val = float(parts[2])  # Temperature value column
            else:
                continue
            
            timestamps.append(ts)
            values.append(val)

        except (ValueError, IndexError):
            continue

# Ensure we found data before plotting
if timestamps:
    # Convert lists to numpy arrays for element-wise operations
    timestamps = np.array(timestamps)
    values = np.array(values)
    
    # Calculate elapsed time in minutes from the start of the file
    minutes = (timestamps - timestamps[0]) / 60.0
    
    # Create the plot
    plt.plot(minutes, values, label='Temperature')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature / Value')
    plt.title('Temperature of Organoids in Scanner')
    plt.grid(True)
    plt.ylim(27,32)
    plt.show()
else:
    print("No valid data was found.")