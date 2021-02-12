import spacy
import re
import numpy as np
import math
# import pickle # use for saving and loading variables calculated by function step2_indexing,
#               # so that in the next time there will be no need to run step2_indexing again


def alphabet_validate(token):
    for i in token:
        if not ( i >= u'\u0061' and i <= u'\u007A'):
            return False
    return True

def step1_preprocessing():
    file = open("./Trec_microblog11.txt", mode='r', encoding='UTF-8-sig')
    sp = spacy.load('en_core_web_sm')
    token_array = []
    line = file.readline()
    while line:
        temp = []
        sentence = sp(line)
        for token in sentence:
            if not token.is_stop and not token.is_punct and not token.is_space and token.text[:7]!='http://':
                if not alphabet_validate(token.lower_):
                    result = re.findall(r'[a-z]+', token.lower_)
                    for i in result:
                        if len(i) > 2:
                            temp.append(i.lower())
                    continue
                if len(token.text) > 2:
                    temp.append(token.lemma_.lower())
        temp.insert(0, line.split()[0])
        token_array.append(temp)
        line = file.readline()
    file.close()
    return(token_array)



def step2_indexing(token_array):
    corpus_size = len(token_array)
    # build vocabulary list
    vocabulary = []
    for doc in token_array:
        for token in doc[1:]:
            if not token in vocabulary:
                vocabulary.append(token)

    # initialize inverted index
    inverted_index = []
    for term in vocabulary:
        entry = []
        entry.append([term, 0])
        entry.append([])
        inverted_index.append(entry)

    # build inverted index
    doc_count = 0
    max_tf = 0
    for doc in token_array:
        docno = doc[0] # document number
        token_list = doc[1:]
        # calculate the number of occurences of each token in a document by using dictionary
        dict = {}
        for key in token_list:
            dict[key] = dict.get(key, 0) + 1

        # update inverted index by the information of the document
        for key in dict:
            tf = dict[key]
            doc_num_tf_pair = [docno, tf]
            key_index = vocabulary.index(key)
            inverted_index[key_index][1].append(doc_num_tf_pair)
            # update df
            inverted_index[key_index][0][1] += 1
            # update max tf
            if tf > max_tf:
                max_tf = tf

        doc_count += 1
        if doc_count % 1000 == 0:
            print("Finished processing " + str(doc_count) + " documents")

    print("Finished processing " + str(doc_count) + " documents")
    vocabulary_size = len(vocabulary)
    print("\n\nFinished building inverted index,")
    print("The size of vocabulary is:", vocabulary_size)

    return inverted_index, vocabulary, max_tf



def step3_retrieval_and_ranking(token_array, inverted_index, vocabulary, max_tf, doc_docno_search_list, query_token_list):
    # find the limited set of documents that contain at least one of the query words
    related_document_number_list = []
    for query_token in query_token_list:
        query_token_index = vocabulary.index(query_token)
        doc_num_tf_pair_list = inverted_index[query_token_index][1]
        for doc_num_tf_pair in doc_num_tf_pair_list:
            doc_num = doc_num_tf_pair[0]
            if not doc_num in related_document_number_list:
                related_document_number_list.append(doc_num)

    # build limited corpus
    related_document_location_list = []
    for docno in related_document_number_list:
        related_document_location_list.append(doc_docno_search_list.index(docno))

    limited_corpus = []
    for location in related_document_location_list:
        limited_corpus.append(token_array[location])

    # build new vocabulary list
    new_vocabulary = []
    for doc in limited_corpus:
        for token in doc[1:]:
            if not token in new_vocabulary:
                new_vocabulary.append(token)

    # calculate idf list
    corpus_size = len(token_array)
    new_vocabulary_idf_list = []
    for term in new_vocabulary:
        term_index = vocabulary.index(term)
        df = inverted_index[term_index][0][1]
        idf = math.log(corpus_size / df, 2)
        new_vocabulary_idf_list.append(idf)

    # build document-docno search list for the limited corpus
    limited_corpus_doc_docno_search_list = []
    for doc in limited_corpus:
        limited_corpus_doc_docno_search_list.append(doc[0])

    # build tf-idf matrix
    limited_corpus_size = len(limited_corpus)
    new_vocabulary_size = len(new_vocabulary)
    tf_idf_matrix = np.zeros((limited_corpus_size, new_vocabulary_size), dtype=np.float)

    for query_token in query_token_list:
        query_token_index = vocabulary.index(query_token)
        doc_num_tf_pair_list = inverted_index[query_token_index][1]
        column_num = new_vocabulary.index(query_token)
        idf = new_vocabulary_idf_list[column_num]
        for doc_num_tf_pair in doc_num_tf_pair_list:
            docno = doc_num_tf_pair[0]
            tf = doc_num_tf_pair[1]
            row_num = limited_corpus_doc_docno_search_list.index(docno)
            # normalize term frequency (tf) across the entire corpus
            # and here we put the process of building term frequency matrix and
            # tf-idf matrix together, to save the time for processing each query
            tf_idf_matrix[row_num][column_num] = (tf / max_tf) * idf

    # # multiply the tf scores by the idf values of each term,
    # # obtaining the following tf-idf matrix
    # tf_idf_matrix = np.zeros((limited_corpus_size, new_vocabulary_size), dtype=np.int)
    # for column_num in range(new_vocabulary_size):
    #     idf = new_vocabulary_idf_list[column_num]
    #     for row_num in range(limited_corpus_size):
    #         tf = tf_matrix[row_num][column_num]
    #         tf_idf_matrix[row_num][column_num] = tf * idf


    # calculate the tf-idf vector for the query
    query_tf_idf_vector = np.zeros((1, new_vocabulary_size), dtype=np.float)
    # first calculate the number of occurences of each token in the query by using dictionary
    dict = {}
    for key in query_token_list:
        dict[key] = dict.get(key, 0) + 1
    # find the maximum token frequency of the query
    query_max_tf = 0
    for key in dict:
        tf = dict[key]
        if tf > query_max_tf:
            query_max_tf = tf
    # calculate the query's tf-idf vector
    for key in dict:
        tf = dict[key]
        key_index = new_vocabulary.index(key)
        idf = new_vocabulary_idf_list[key_index]
        # a modified tf-idf weighting scheme w_iq = (0.5 + 0.5 tf_iq)∙idf_i
        query_tf_idf_vector[0][key_index] = (0.5 + 0.5 * (tf / query_max_tf)) * idf
        # # traditional tf-idf weighting scheme
        # query_tf_idf_vector[0][key_index] = (tf / query_max_tf) * idf

    # compute the similarity scores between a query and each document using cosine
    # save the result into a dictionary
    similarity_dictionary = {}
    query_tf_idf_vector_length = np.linalg.norm(query_tf_idf_vector)
    query_tf_idf_vector = query_tf_idf_vector.flatten()
    for row_num in range(limited_corpus_size):
        docno = limited_corpus_doc_docno_search_list[row_num]
        document_tf_idf_vector = tf_idf_matrix[row_num]
        cosSim = sum(document_tf_idf_vector * query_tf_idf_vector) / (np.linalg.norm(document_tf_idf_vector) * query_tf_idf_vector_length)
        similarity_dictionary[docno] = cosSim

    return similarity_dictionary



# small test case

token_array = step1_preprocessing()
inverted_index, vocabulary, max_tf = step2_indexing(token_array)

# build document-docno search list
doc_docno_search_list = []
for doc in token_array:
    doc_docno_search_list.append(doc[0])


# example query token list
example_query_token_list = ['bbc', 'world', 'service', 'staff', 'cuts']

sim_dict = step3_retrieval_and_ranking(token_array, inverted_index,
                                       vocabulary, max_tf, doc_docno_search_list, example_query_token_list)






