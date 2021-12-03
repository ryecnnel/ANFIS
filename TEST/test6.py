import pandas as pd

csv_teststock_input = pd.read_csv(filepath_or_buffer="./Dataset/TEST_STOCK.csv", encoding="utf-8", sep=",")
#print(csv_teststock_input.values)

csv_xor_input = pd.read_csv(filepath_or_buffer="./Dataset/XOR.csv", encoding="utf-8", sep=",")
#print(csv_xor_input.values)

X_train = csv_xor_input.values[:,:2]
t = csv_xor_input.values[:,2]
print(X_train, t)