# results_plot.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_results():
    try:
        y_true = pd.read_csv('data/test_target.csv').values.ravel()
        y_pred = pd.read_csv('data/submission.csv').values.ravel()
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.savefig('visualize_results/plot_metric.png')
        plt.show()
    except Exception as e:
        logger.info(f"Error at plotting step: {e}")

if __name__ == '__main__':
    plot_results()
