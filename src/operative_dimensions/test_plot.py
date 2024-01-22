import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual data)
n_op_dims = 100
a, b = 0.1, 0.3
# Parameters
trend_factor_1 = np.random.uniform(a, b, size=n_op_dims)  # Adjust the trend factor as needed
trend_factor_2 = np.random.uniform(a, b, size=n_op_dims)  # Adjust the trend factor as needed

# Generate arrays with trends
array_1 = np.linspace(0, 1, n_op_dims) + trend_factor_1 * np.arange(n_op_dims) + np.random.normal(scale=0.02, size=n_op_dims)
array_2 = np.linspace(0, 1, n_op_dims) + trend_factor_2 * np.arange(n_op_dims) + np.random.normal(scale=0.02, size=n_op_dims)

# Reshape arrays to (n_op_dims, 1)
mses_1 = array_1.reshape(-1, -1)
mses_2 = array_2.reshape(-1, 1)
x_data = np.arange(100)

# Parameters
my_fontsize = 12
my_xlabel = "X Label"
my_ylabel = "Y Label"
y_max = 1.5  # Adjust as needed

fig, ax = plt.subplots(1, 1)
plt.grid()

display_names = ["Name1", "Name2"]
if display_names is None:
    display_name1 = ''
    display_name2 = ''
else:
    display_name1 = display_names[0]
    display_name2 = display_names[1]

ax.plot(x_data, y_data_mean[0, :], "r", linewidth=3, label=display_name1)
ax.plot(x_data, y_data2_mean[0, :], "b", linewidth=3, label=display_name2)

ax.fill_between(x_data, y_data_mean[0, :] - y_data_var[0, :], y_data_mean[0, :] + y_data_var[0, :],
                color='red', alpha=0.3)
ax.fill_between(x_data, y_data2_mean[0, :] - y_data2_var[0, :], y_data2_mean[0, :] + y_data2_var[0, :],
                color='b', alpha=0.3)

# labels
ax.tick_params(axis='both', which='major', labelsize=my_fontsize)
ax.set_xlabel(my_xlabel, fontsize=my_fontsize)
ax.set_ylabel(my_ylabel, fontsize=my_fontsize)

if not (display_names is None):
    plt.legend(fontsize=my_fontsize)

ax.set_ylim(bottom=0, top=1.1 * y_max)

plt.show()