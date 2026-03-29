# unsupervised-learning-lab

## Project Overview

This project is a data science analysis focused on unsupervised machine learning. It uses clustering algorithms, specifically K-Means and DBSCAN, to analyze the New York City Airbnb Open Data dataset. The primary goal is to identify distinct groups or segments within the Airbnb listings based on their features.

The analysis is implemented in Python and leverages several key data science libraries:
- **`pandas`** and **`numpy`** for data manipulation and numerical operations.
- **`scikit-learn`** for implementing the K-Means and DBSCAN clustering algorithms, as well as for data preprocessing (PCA) and evaluation metrics.
- **`matplotlib`** and **`seaborn`** for data visualization and generating charts to interpret the clustering results.

## Key Files

-   `unsupervised_learning.py`: The main Python script that executes the entire analysis pipeline. It loads the data, performs clustering, evaluates the results, and saves all output visualizations.
-   `unsupervised_learning.ipynb`: A Jupyter Notebook version of the analysis, likely used for interactive development and exploration.
-   `preprocess_data.csv`: The raw or minimally processed dataset.
-   `postprocess_data.csv`: The cleaned, scaled, and feature-engineered dataset used as the primary input for the machine learning models.
-   `charts/`: A directory where all the generated visualizations (histograms, scatter plots, radar charts, etc.) are saved.
-   `unsupervised_learning_report.pdf`: A PDF document that likely contains a detailed report and summary of the findings from the analysis.

## Building and Running

### Dependencies

To run this project, you need Python and the following libraries installed. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn reportlab
```

### Running the Analysis

The entire analysis can be executed by running the main Python script from the command line:

```bash
python unsupervised_learning.py
```

This command will:
1.  Load the data from the `.csv` files.
2.  Perform K-Means and DBSCAN clustering.
3.  Print evaluation metrics to the console.
4.  Generate and save all analytical charts into the `charts/` directory.

