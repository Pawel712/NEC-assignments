import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('loss_data_dataset2.csv')

plt.scatter(data['# FirstLoss'], data['SecondLoss'], color='black', alpha=0.5)
plt.gcf().set_facecolor('lightgreen')  


plt.xlabel('First loss')
plt.ylabel('Second loss')
plt.title('Scatter plot')

plt.show()
