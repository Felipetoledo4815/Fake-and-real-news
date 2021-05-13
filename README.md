
# Classifying Fake and Real News
This is the repository to accompany the CS 6501-007 Natural Language Processing final project. In response to the call for [NLP for Positive Impact](https://sites.google.com/view/nlp4positiveimpact2021), we performed text-classification on a dataset of [fake/real news](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

## Data pre-processing
All the code for this section is inside `data_processing.ipynb` notebook. To execute it, you will need to download the original `Fake.csv` and `Real.csv` datasets from Kaggle. However, if you want to get the already pre-processed and splitted datasets, go to the `Download preprocessed data` section inside the jupyter-notebook and execute the cell, which will download the data and put it in a folder which will later be used.
## Embeddings
In this section we implemented different embeddings for the pre-processed dataset created before. Each embedding is in a different jupyter-notebook file.
### BoW
All the code is inside the `embeddings/BoW_embedding.ipynb` notebook. This jupyter-notebook should be executed completely to create and save the embeddings that will later be used.
### GloVe
### Word2Vec (Skipgram)
### ELMo
All the code is inside the `embeddings/ELMo_embedding.ipynb` notebook. The first section `Load ELMo` will download the weights of the pre-trained ELMo model created by AllenNLP and save them into `downloads` folder.

Then the notebook has two major sections `Titles embeddings` and `Texts embeddings`, which contain the code to generate the ELMo embeddings for the titles and the texts respectively. It is worth noting that creating the embedding is an intensive task thus a GPU usage is recommended. For the seek of simplicity, we uploaded to google drive the embeddings for the titles only since its weight was around 4 gb for the 3 splits. Unfortunately we were not able to upload the titles embeddings since its weight was around 15 gb for the 3 splits. If you would like to download the titles embedding you can do that by running the cell under the `Download titles preprocessed embeddings`.
## Models
In this section, we compared the different titles embeddings using a simple logistic regression classifier. And then we used ELMo embedding for the titles to train the different Neural Network models. In addition we fine-tunned the BERT model using BERT embeddings for our dataset and for completeness, we also fine-tunned it using the ELMo embeddings for the texts.
### Logistic Regression
All the code is inside the `models/linear_classifier.ipynb` notebook. Here we loaded the different embeddings described above and use them with the loggistic regression model from Scikit Learn. We calculated the training and test accuracy, as well as precision, recall and F-1 score for each embedding.
### CNN
All the code is inside the `models/CNN.ipynb` notebook. Here we implemented a very simple CNN and trained it using the ELMo embeddings for the titles. We computed training, validation and test accuract, as well as precision, recall and F-1 score.
### RNN
### BERT

## App - Chrome Extension
Chrome Extension is available in the CX directory. Steps to use the chrome extension are listed below:
* Launch the Flask Server with the classifier in the cx directory using `python server.py`
* In your Chrome Browser navigate to **chrome://extensions**
* Select **Load Unpacked**
* Select the extension subdirectory in cx 

The extension should be available in Chrome.


## Embeddings

### GloVe

The GloVe pretrained embeddings are available [here](http://nlp.stanford.edu/data/glove.840B.300d.zip):  
Unzip the archive and move the `glove.840B.300d.txt` to the embeddings folder.





