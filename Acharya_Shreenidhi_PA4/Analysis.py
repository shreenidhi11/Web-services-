# Author: Shreenidhi
# Description: Implement web service classification and
# clustering using programmable web data (api.txt)

# import statements
import faulthandler
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, pairwise_distances, silhouette_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
# download the nlp functionality
nltk.download('stopwords')
nltk.download('punkt')

def read_api_data(filename):
    """
    Read the API data from the file
    input: The file to read
    return: The JSON object containing the API data
    """
    api_records = []
    with open(filename, 'r', encoding="ISO-8859-1") as f:
        lines = f.readlines()

        for line in lines:
            fields = line.lower().strip().split('$#$')
            rating = float(fields[3]) if fields[3] else 0.0
            api_record = {
                'id': fields[0],
                'title': fields[1],
                'summary': fields[2],
                'rating': rating,
                'name': fields[4],
                'label': fields[5],
                'author': fields[6],
                'description': fields[7],
                'type': fields[8],
                'downloads': fields[9],
                'useCount': fields[10],
                'sampleUrl': fields[11],
                'downloadUrl': fields[12],
                'dateModified': fields[13],
                'remoteFeed': fields[14],
                'numComments': fields[15],
                'commentsUrl': fields[16],
                'tags': " ".join(fields[17].split('###')),
                'category': fields[18],
                'protocols': fields[19],
                'serviceEndpoint': fields[20],
                'version': fields[21],
                'wsdl': fields[22],
                'dataFormats': fields[23],
                'apiGroups': fields[24],
                'example': fields[25],
                'clientInstall': fields[26],
                'authentication': fields[27],
                'ssl': fields[28],
                'readonly': fields[29],
                'VendorApiKits': fields[30],
                'CommunityApiKits': fields[31],
                'blog': fields[32],
                'forum': fields[33],
                'support': fields[34],
                'accountReq': fields[35],
                'commercial': fields[36],
                'provider': fields[37],
                'managedBy': fields[38],
                'nonCommercial': fields[39],
                'dataLicensing': fields[40],
                'fees': fields[41],
                'limits': fields[42],
                'terms': fields[43],
                'company': fields[44],
                'updated': fields[45],
                # New field with features for every service
                'service_category': fields[7] +fields[2] + " ".join(fields[17].split('###')) 
            }

            api_records.append(api_record)
    return api_records

def load_api_data(data):
    '''
    Select the features needed(X) and label (Y)
    input: Original dataframe containing all the columns
    return: New dataframe containing selected features and labels
    '''
    # take only related fields
    api_data = pd.DataFrame(data)
    api_data = api_data[['service_category',
                         'tags', 'name', 'title', 'category']]
    return api_data


def clean_api_data(loaded_api_data):
    '''
    Clean the dataframe by removing duplicate rows and null values 
    input: Dataframe with only features and labels
    return: Cleaned dataframe
    '''
    # remove duplicates
    data_without_duplicates = loaded_api_data.drop_duplicates()
    # remove null values
    data_without_null_values = data_without_duplicates.dropna()
    return data_without_null_values

# Define a function to remove stopwords
def remove_stopwords(tokens):
    '''
    Removes stopwords like is, the, these
    input: tokens
    return: filtered tokens
    '''
    stop_words_list = set(stopwords.words('english')) 
    filtered_tokens = [
        word for word in tokens if word.lower() not in stop_words_list]
    return filtered_tokens

def remove_stop_words(cleaned_api_data):
    '''
    Removes stopwords from the service category column
    input: Dataframe with only features and labels
    return: service category without any stopwords
    '''
    # tokenize
    cleaned_api_data['service_category'] = cleaned_api_data['service_category'].apply(
        word_tokenize)
    # remove the stopwords
    cleaned_api_data['service_category'] = cleaned_api_data['service_category'].apply(
        remove_stopwords)
    # join the new sentences post removal
    cleaned_api_data['service_category'] = cleaned_api_data['service_category'].apply(
        ' '.join)
    return cleaned_api_data


def balancing_api_data(api_records):
    '''
    balancing the api data records
    input: cleaned api data
    return: balanced api data
    '''
    new_api_records = api_records['category'].value_counts().to_dict()
    sorted_records = dict(sorted(new_api_records.items(),
                          key=lambda item: item[1], reverse=True))
    count_of_highest_category = list(sorted_records.values())[0]
    threshold_value = count_of_highest_category * 0.3
    balanced_api_records_category = []

    # take only common values
    for key in sorted_records:
        if sorted_records[key] >= threshold_value:
            balanced_api_records_category.append(key)
        else:
            break
    
    # create a new data frame with balanced categories
    balanced_api_records = api_records[api_records['category'].isin(
        balanced_api_records_category)]
    return balanced_api_records

def kmeans_clustering_tfidf(data):
    '''
    Perform kmeans clustering with TFIDF model
    input: api records
    '''
    new_record = data['service_category'].values.astype('U')
    tfidf_vector = TfidfVectorizer()
    features = tfidf_vector.fit_transform(new_record)
    kmeans = KMeans(n_clusters=50, 
                init='k-means++', 
                n_init=1, 
                max_iter=100, 
                tol=1e-4, 
                random_state=0)

    kmeans.fit(features)
    # creates a new column for labels
    data['clusters'] = kmeans.labels_
    # calculate the silhouette score as part of metric
    silhouette_avg = silhouette_score(features, kmeans.labels_)
    print("TFIDF with Silhouette Score:", silhouette_avg)

def kmeans_clustering_lda(data):
    '''
    Perform kmeans clustering with LDA model
    input: api records
    '''
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(data['service_category'].values.astype('U'))
    lda = LatentDirichletAllocation(n_components=13, random_state=42)
    list_of_topics= lda.fit_transform(X_tfidf)

    kmeans = KMeans(n_clusters=50, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                tol=1e-4, 
                random_state=0)
    
    kmeans.fit(list_of_topics)
    data['clusters'] = kmeans.labels_
    # Calculate Silhouette Score
    silhouette = silhouette_score(list_of_topics,  kmeans.labels_)
    print("LDA with Silhouette Score:", silhouette)

def kmeans_clustering_word_embed(data):
    '''
    Perform kmeans clustering with word embed model
    input: api records
    '''
    # encoding each record from the input api record
    sentences = [records.split() for records in data['service_category'].values.astype('U')]
    word_embed_model = Word2Vec(sentences, window=5, min_count=1, epochs=10)
    sentence_vectors = []
    # find the vector
    for sentence in range(len(sentences)):
        sum_of_vector = np.zeros(shape=word_embed_model.vector_size)
        for word in sentences[sentence]:
            if word in word_embed_model.wv:
                sum_of_vector+=word_embed_model.wv[word]
        average = sum_of_vector / len(sentences[sentence])
        sentence_vectors.append(average)

    kmeans = KMeans(n_clusters=50, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                tol=1e-4, 
                random_state=0)
    
    kmeans.fit(sentence_vectors)
    
    # Predict cluster labels
    predicted_labels = kmeans.predict(sentence_vectors)
    
    # Calculate Silhouette Score
    silhouette = silhouette_score(sentence_vectors, kmeans.labels_)
    print("Word Embed with Silhouette Score:", silhouette)

def kmeans_clustering_bert(data):
    '''
    Perform kmeans clustering with BERT model
    input: api records
    '''
    result = []
    # Found the embedding for only 1000 record due to storage issue
    for category in data[0:1000]['service_category']:
        category_embedding = get_BERT_Embedding(category)
        result.append(category_embedding)
    result = np.array(result)
    # flattening the result 
    result_flattened = result.reshape(result.shape[0], -1)

    kmeans = KMeans(n_clusters=2, 
            init='k-means++', 
            n_init=10, 
            max_iter=300, 
            tol=1e-4, 
            random_state=0)
    
    cluster_labels = kmeans.fit_predict(result_flattened)
    silhouette_avg = silhouette_score(result_flattened, kmeans.labels_)
    print("BERT with Silhouette Score", silhouette_avg)

def clustering_api_records_kmeans(data):
    '''
    Main function that calls kmeans clustering on all the models
    input: api records
    '''
    print("Kmeans Clustering\n")
    kmeans_clustering_tfidf(data)
    kmeans_clustering_lda(data)
    kmeans_clustering_word_embed(data)
    kmeans_clustering_bert(data)

def clustering_api_records_dbscan(data):
    '''
    Main function that calls DBSCAN clustering on all the models
    input: api records
    '''
    print("DBSCAN Clustering\n")
    dbscan_clustering_tfidf(data)
    dbscan_clustering_lda(data)
    dbscan_clustering_word_embed(data)
    dbscan_clustering_bert(data)

def dbscan_clustering_tfidf(data):
    '''
    Perform DBSCAN clustering with TFIDF model
    input: api records
    '''
    # Encode the records 
    new_record = data['service_category'].values.astype('U')
    tfidf_vector = TfidfVectorizer()
    features = tfidf_vector.fit_transform(new_record)

    # Create the model and fit 
    dbscan = DBSCAN(eps=1.0, min_samples=10)
    clusters = dbscan.fit_predict(features)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(features, clusters)
    print("TFIDF with Silhouette Score", silhouette_avg)

def dbscan_clustering_lda(data):
    '''
    Perform DBSCAN clustering with LDA model
    input: api records
    '''
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(data['service_category'])
    lda = LatentDirichletAllocation(n_components=13, random_state=0)
    list_of_topics = lda.fit_transform(X_tfidf)
    cos_sim_matrix = cosine_similarity(list_of_topics)
    dbscan = DBSCAN(eps=0.1, min_samples=3)
    clusters = dbscan.fit_predict(cos_sim_matrix)

    silhouette_avg = silhouette_score(cos_sim_matrix, clusters)
    print("LDA with Silhouette Score", silhouette_avg)

def  dbscan_clustering_word_embed(data):
    '''
    Perform DBSCAN clustering with Word Embed model
    input: api records
    '''
    # taking only the starting 1000 records
    sentences = [records.split() for records in data['service_category']]
    word_embed_model = Word2Vec(sentences[:1000], window=5, min_count=1, epochs=10)
    # word_embed_model = Word2Vec(sentences, window=5, min_count=1, epochs=10)
    vocab = word_embed_model.wv.index_to_key
    word_vectors = [word_embed_model.wv[word] for word in vocab]

    cos_sim_matrix = np.abs(cosine_similarity(word_vectors))
    dbscan = DBSCAN(eps=3.0, min_samples=3)
    clusters = dbscan.fit_predict(cos_sim_matrix)
    silhouette = silhouette_score(cos_sim_matrix, clusters)
    print("Word Embed with Silhouette Score:", silhouette)

def dbscan_clustering_bert(data):
    '''
    Perform DBSCAN clustering with BERT model
    input: api records
    '''
    # taking only the starting 1000 records
    result = []
    for category in data[0:1000]['service_category']:
        category_embedding = get_BERT_Embedding(category)
        result.append(category_embedding)
    result = np.array(result)
    result_flattened = result.reshape(result.shape[0], -1)
    dbscan = DBSCAN(eps=3.0, min_samples=3)

    cluster_labels = dbscan.fit_predict(cosine_similarity(result_flattened))

    # Since i took only 1000 redcords, the dbscan was able tp generate only 1 label
    # hence check for that
    if len(set(dbscan.labels_))== 1:
      print("DBSCAN for BERT Model was able to find only 1 cluster")
    else:
      silhouette_avg = silhouette_score(result_flattened, cluster_labels)
      print("BERT with Silhouette Score", silhouette_avg)

def clustering_api_records(data):
    '''
    Perform kmeans and dbscan on all the models
    input: api records
    '''
    clustering_api_records_kmeans(data)
    clustering_api_records_dbscan(data)

def get_BERT_Embedding(inputText):
    '''
    Find the BERT Embedding for each vector
    input: api records
    return: sentence embeddings
    '''
    # Load BERT tokenizer and model
    # this code is taken from the PPT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    encoding = tokenizer.batch_encode_plus(
      [inputText],                    # List of input texts
      padding=True,              # Pad to the maximum sequence length
      truncation=True,           # Truncate to the maximum sequence length if necessary
      return_tensors='pt',      # Return PyTorch tensors
      add_special_tokens=True    # Add special tokens CLS and SEP
    )
    input_ids = encoding['input_ids']  # Token IDs
    attention_mask = encoding['attention_mask']  # Attention mask
    with torch.no_grad():     # Generate embeddings using BERT model
      outputs = model(input_ids, attention_mask=attention_mask)
      word_embeddings = outputs.last_hidden_state  # This contains the embeddings
    sentence_embedding = word_embeddings.mean(dim=1)  # Average pooling along the sequence length dimension
    return sentence_embedding

def classify_api_records(data):
    '''
    Perform text classification on TFIDF, LDA, Word Embed and BERT
    and then used classification algorithms like decision tree, navies bayesian
    and K nearest neighbours
    input: api records
    '''
    # divide the data into training and testing.
    x_training, x_testing, y_training, y_testing = train_test_split(data['service_category'], data['category'],
                                                                    test_size=0.2, random_state=42)
    tf_idf_model = TfidfVectorizer()
    lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
    
    # defining all the classifiers here
    classifier_1 = DecisionTreeClassifier()
    classifier_2 = MultinomialNB()
    classifier_3 = KNeighborsClassifier(n_neighbors=15)

    # tfidf
    X_train_tfidf = tf_idf_model.fit_transform(x_training)
    X_test_tfidf = tf_idf_model.transform(x_testing)

    # lda
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(x_training)
    X_train_lda = lda_model.fit_transform(X_train_counts)
    X_test_counts = vectorizer.transform(x_testing)
    X_test_lda = lda_model.transform(X_test_counts)

    # word embedding
    sentences = [records.split() for records in data['service_category']]
    word_embed_model = Word2Vec(sentences, window=5, min_count=1, epochs=10)
    X_word = [sum(word_embed_model.wv[word]
                  for word in words) / len(words) for words in sentences]
    X_word_abs = [np.abs(embedding) for embedding in X_word]
    x_word_train, x_word_test, y_word_train, y_word_test = train_test_split(X_word_abs, data['category'], test_size=0.2,
                                                                            random_state=42)
    # bert
    result = []
    for category in data[0:1000]['service_category']:
        category_embedding = get_BERT_Embedding(category)
        result.append(category_embedding)
    result = np.array(result)
    result_flattened = result.reshape(result.shape[0], -1)
    X_word_abs_ = [np.abs(embedding) for embedding in result_flattened]
    labels = data[0:1000]['category']
    X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(X_word_abs_, labels, test_size=0.2, random_state=42)

    #for model tfidf
    # for classifier 1
    print("TFIDF - Decision Tree Classifier")
    classifier_1.fit(X_train_tfidf, y_training)
    y_predicted_DT = classifier_1.predict(X_test_tfidf)
    scores_1_DT = cross_val_score(
        classifier_1, X_train_tfidf, y_training, cv=5)
    print("\tCross-validation scores for tfidf:", scores_1_DT.mean())
    print("\tAccuracy for tfidf: ", accuracy_score(y_testing, y_predicted_DT))
    print("\n")

    # for classifier 2
    print("TFIDF - Naive Bayesian Classifier")
    classifier_2.fit(X_train_tfidf, y_training)
    y_predicted_NB = classifier_2.predict(X_test_tfidf)
    scores_1_NB = cross_val_score(
        classifier_2, X_train_tfidf, y_training, cv=5)
    print("\tCross-validation scores for tfidf:", scores_1_NB.mean())
    print("\tAccuracy for tfidf: ", accuracy_score(y_testing, y_predicted_NB))
    print("\n")

    # for classifier 3
    print("TFIDF - KNN Classifier")
    classifier_3.fit(X_train_tfidf, y_training)
    y_predicted_KNN = classifier_3.predict(X_test_tfidf)
    scores_1_KNN = cross_val_score(
        classifier_3, X_train_tfidf, y_training, cv=5)
    print("\tCross-validation scores for tfidf:", scores_1_KNN.mean())
    print("\tAccuracy for tfidf: ", accuracy_score(y_testing, y_predicted_KNN))
    print("\n")

    # for model lda
    # for classifier 1
    print("LDA - Decision Tree Classifier")
    classifier_1.fit(X_train_lda, y_training)
    y_predicted_DT = classifier_1.predict(X_test_lda)
    scores_1_DT = cross_val_score(classifier_1, X_train_lda, y_training, cv=5)
    print("\tCross-validation scores for lda:", scores_1_DT.mean())
    print("\tAccuracy for lda: ", accuracy_score(y_testing, y_predicted_DT))
    print("\n")

    # for classifier 2
    print("LDA - Naive Bayesian Classifier")
    classifier_2.fit(X_train_lda, y_training)
    y_predicted_NB = classifier_2.predict(X_test_lda)
    scores_1_NB = cross_val_score(classifier_2, X_train_lda, y_training, cv=5)
    print("\tCross-validation scores for lda:", scores_1_NB.mean())
    print("\tAccuracy for lda: ", accuracy_score(y_testing, y_predicted_NB))
    print("\n")

    # for classifier 3
    print("LDA - KNN Classifier")
    classifier_3.fit(X_train_lda, y_training)
    y_predicted_KNN = classifier_3.predict(X_test_lda)
    scores_1_KNN = cross_val_score(classifier_3, X_train_lda, y_training, cv=5)
    print("\tCross-validation scores for lda:", scores_1_KNN.mean())
    print("\tAccuracy for lda: ", accuracy_score(y_testing, y_predicted_KNN))
    print("\n")

    # for model word embedding
    # for classifier 1
    print("Word Embedding - Decision Tree Classifier")
    classifier_1.fit(x_word_train, y_word_train)
    y_predicted_DT = classifier_1.predict(x_word_test)
    scores_1_DT = cross_val_score(classifier_1, X_word, data['category'], cv=5)
    print("\tCross-validation scores for word embedding:", scores_1_DT.mean())
    print("\tAccuracy for word embedding: ",
          accuracy_score(y_word_test, y_predicted_DT))
    print("\n")

    # for classifier 2
    print("Word Embedding - Naive Bayesian Classifier")
    classifier_2.fit(x_word_train, y_word_train)
    y_predicted_NB = classifier_2.predict(x_word_test)
    scores_1_NB = cross_val_score(classifier_2, X_word_abs, data['category'], cv=5)
    # scores_1_NB = cross_val_score(classifier_2, X_word_abs, data['category'], cv=5)
    print("\tCross-validation scores for word embedding:", scores_1_NB.mean())
    print("\tAccuracy for word embedding: ",
          accuracy_score(y_word_test, y_predicted_NB))
    print("\n")

    # for classifier 3
    print("Word Embedding - KNN Classifier")
    classifier_3.fit(x_word_train, y_word_train)
    y_predicted_KNN = classifier_3.predict(x_word_test)
    scores_1_KNN = cross_val_score(
        classifier_3,  X_word, data['category'], cv=5)
    print("\tCross-validation scores for word embedding:", scores_1_KNN.mean())
    print("\tAccuracy for word embedding: ",
          accuracy_score(y_word_test, y_predicted_KNN))
    print("\n")

    # for model BERT
    # for classifier 1
    print("BERT Embedding - Decision Tree Classifier")
    classifier_1.fit(X_train_bert, y_train_bert)
    y_predicted = classifier_1.predict(X_test_bert)
    scores_1_DT = cross_val_score(classifier_1,X_train_bert, y_train_bert, cv=5)
    print("\tCross-validation scores for bert embedding:", scores_1_DT.mean())
    print("\tAccuracy for bert embedding: ",
          accuracy_score(y_test_bert, y_predicted))
    
    # for classifier 2
    print("BERT Embedding - Naive Bayesian Classifier")
    classifier_2.fit(X_train_bert, y_train_bert)
    y_predicted = classifier_2.predict(X_test_bert)
    scores_1_NB = cross_val_score(classifier_2, X_train_bert, y_train_bert, cv=5)
    print("\tCross-validation scores for bert embedding:", scores_1_NB.mean())
    print("\tAccuracy for bert embedding: ",
    accuracy_score(y_test_bert, y_predicted))
    
    # for classifier 3
    # made the value ok k=2 just for the purpose of testing!
    print("BERT Embedding - KMeans Classifier")
    classifier_3.fit(X_train_bert, y_train_bert)
    y_predicted = classifier_3.predict(X_test_bert)
    scores_1_KNN = cross_val_score(classifier_3, X_train_bert, y_train_bert, cv=5)
    print("\tCross-validation scores for BERT embedding:", scores_1_KNN.mean())
    print("\tAccuracy for BERT embedding: ",
        accuracy_score(y_test_bert, y_predicted))
    
def main():
    # read the record
    api_records = read_api_data(
        "api.txt")
    
    # data cleaning starts
    loaded_api_data = load_api_data(api_records)

    # balancing the api_data for category selection
    balanced_data = balancing_api_data(loaded_api_data)
    cleaned_api_data = clean_api_data(balanced_data)
    cleaned_data = remove_stop_words(cleaned_api_data)
    # data cleaning stops

    # classification
    classify_api_records(cleaned_data)

    # clustering
    clustering_api_records(balanced_data)

if __name__ == "__main__":
    faulthandler.enable()
    main()
