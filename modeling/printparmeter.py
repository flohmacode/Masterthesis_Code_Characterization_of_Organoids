import numpy as np

'''This Script allows to investigate parameters calculated by the ABC-Algorithm'''

leupold_feb = np.load('./modeling/parameters/leupold_feb_linear_best_parameter_numbers.npy',allow_pickle=True)

jojo_april = np.load('./modeling/parameters/jojo_april_linear_best_parameter_numbers.npy',allow_pickle=True)

print(leupold_feb)
print(jojo_april)