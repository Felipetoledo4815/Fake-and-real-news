
# Classifying Fake and Real News
This is the repository to accompany the CS 6501-007 Natural Language Processing final project. In response to the call for [NLP for Positive Impact](https://sites.google.com/view/nlp4positiveimpact2021), we performed text-classification on a dataset of [fake/real news](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

## Data pre-processing
All the code for this section is inside `data_processing.ipynb` notebook. To execute it, you will need to download the original `Fake.csv` and `Real.csv` datasets from Kaggle. However, if you want to get the already pre-processed and splitted datasets, go to the `Download preprocessed data` section inside the jupyter-notebook and execute the cell, which will download the data and put it in a folder which will later be used.
## Embeddings
In this section we implemented different embeddings for the pre-processed dataset created before. Each embedding is in a different jupyter-notebook file.
### BoW
All the code is inside the `embeddings/BoW_embedding.ipynb` notebook. This jupyter-notebook should be executed completely to create and save the embeddings that will later be used.
### GloVe
The GloVe pretrained embeddings are available 
In order to generate the GloVe Embeddings, sumply run the Glove_embedding notebook in the embedding directory. The notebook contains cells to download the embeddings from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip) and will then create the embeddings for the sentences and dump them in the preprocessed_embeddings directory

### Word2Vec (Skipgram)
All the code is inside the `embeddings/skipgram_embedding.ipynb` notebook. All the cells should be executed from top to bottom to get the skipgram embeddings inside the `preprocessed_embeddings` folder.
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
For the comparison of BERT classification accuracy we use the Huggingface [BERTForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification). 

In the case where we use the default BERT embeddings and text from the dataset as input, we use the smaller [BERT-base-uncased](https://huggingface.co/bert-base-uncased) model.  

To use the BERT model with ELMo embedded text as input we use the [BERT-large-uncased](https://huggingface.co/bert-large-uncased) model.


## App - Chrome Extension
The Chrome Extension is available in the `cx` directory. This directory contains a jupyter-notebook called `classifier.ipynb`, which create, train and save a pipeline that includes CountVectorizer function first, and then a logistic regression model. Then the `server.py` has the code for running the Flask server, which host the saved pipeline and expose an endpoint to return the predictions.

The steps to use the chrome extension are listed below:

1. Launch the Flask Server with the classifier in the cx directory using `python server.py`
2. In your Chrome Browser navigate to **chrome://extensions**
3. Select **Load Unpacked**
4. Select the extension subdirectory in cx 

The extension should be available in Chrome. To use it, simply open a news and click the extension if you have doubts wheter it is real or fake. If the extension does not show anything, then it means that the news is real, otherwise it will show a warning sign.
