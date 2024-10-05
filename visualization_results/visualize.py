# results_plot.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_results():
    try:
        y_true = pd.read_csv('iris_target.csv').values.ravel()
        y_pred = pd.read_csv('submission.csv').values.ravel()
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.savefig('plot_metric.png')
        plt.show()
    except Exception as e:
        print(f"Error at plotting step: {e}")

if __name__ == '__main__':
    plot_results()
