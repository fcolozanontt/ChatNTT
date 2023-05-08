print('We begin by importing a csv containing the rows we must predict')
import csv
file = open("test.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
for index in range(len(data[0])):
    data[0][index] = int(data[0][index])
print(data)

print('Now, we open our model and predict it')
import pickle 
from sklearn.neighbors import KNeighborsClassifier
loaded_model = pickle.load(open('knnpickle_file', 'rb'))
print(f'loaded_model.predict({data})'.format(data=data), loaded_model.predict(data) )