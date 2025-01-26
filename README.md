# Predicting Citations Between Scientific Publications

## Dataset
The Cora dataset consists of **2708 scientific publications** classified into one of **seven classes**. The citation network consists of **5429 links**. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. (*taken from [Keggle](https://paperswithcode.com/dataset/cora)*)
### Data Visualization
The `data_prep.py` script contains methods for plotting the graph.

![Cora dataset plot](https://github.com/lkuresevic/sci-pub-link-prediction/blob/main/dataset_plot.png)
## Testing
After installing necessary dependencies from `requirements.txt`, a model can be trained and evaluated by running `python3 main.py` from the command line.
