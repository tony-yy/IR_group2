from pyserini import collection, index
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
# import random
import json
import numpy as np
# from nltk.stem.snowball import SnowballStemmer
from functools import reduce
from sklearn import datasets as ds
import pandas as pd
from scipy.sparse.csc import csc_matrix
import csv
from tqdm import tqdm
import re
import random
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from bm25_passage_rank import set_bm25_parameters

# print("import finished")
# collection paths:
collection_name = 'JsonCollection'
collection_dir = 'D:/MSMARCO/passage_retrieval/raw/collection/msmarco-passage/collection_jsonl/'
index_dir = 'indexes'
docid_list_dir = 'features/docid_list.json'

searcher = SimpleSearcher(index_dir)
index_reader = IndexReader(index_dir)
# print("initialize indexer finished")

'''
directories
'''
# feature storage path:
train_feats_dir = 'features/train_feats.txt'    # 1. training set features
test_feats_dir = 'features/test_feats.txt'      # 2. preprocessed test features

# training examples:
train_label_dir = "D:/MSMARCO/passage_retrieval/raw/queries/qrels.train.tsv"    # 1. train qrel file path
train_query_dir = "D:/MSMARCO/passage_retrieval/raw/queries/queries.train.tsv"  # 2. train query file path
train_qids = "features/baseline_features/train_qids.txt"      # 3. train qrel entry qid storage path
train_docids = "features/baseline_features/train_docids.txt"  # 4. train qrel entry docid storage path
group_dir = "features/group.txt"            # 5. train qrel entry group storage path

doclen_ave = index_reader.stats()['total_terms'] / index_reader.stats()['documents']
# print(doclen_ave)

def get_docid_list(collection_class: str, collection_dir: str, dir=None):
    docid = []
    coll = collection.Collection(collection_class, collection_dir)
    generator = index.Generator('DefaultLuceneDocumentGenerator')
    for (i, fs) in enumerate(coll):
        for (j, doc) in enumerate(fs):
            parsed = generator.create_document(doc)
            docid.append(parsed.get('id')) 
    if dir:
        with open(dir, 'w') as f:
            json.dump(docid, f)
    return docid

def get_docid_list_from_file(data_path):
    with open(data_path) as f:
        docid = json.load(f)
    # print(type(docid))
    return docid

# query-doc features
def tf_t_d(docid: str, term:str):
    doc_vec = index_reader.get_document_vector(docid)
    token = index_reader.analyze(term)[0]
    tf_t_d_score = 0.5 if token not in doc_vec else doc_vec[token]
    return tf_t_d_score
def tf_doc(docid: str):
    return index_reader.get_document_vector(docid)

def df_doc(docid: str):
    return {term: (index_reader.get_term_counts(term, analyzer=None))[0] for term in tf_doc(docid).keys()}
def df_t(term: str):
    df, _ = index_reader.get_term_counts(term)
    df = 0.5 if df==0 else df
    return df

def tfidf_t_d(docid: str, term: str):
    num_docs = index_reader.stats()['documents']   
    return tf_t_d(docid, term)/doc_len(docid) * np.log(num_docs/(df_t(term)+1))
def ictf_t_d(docid: str, term: str):
    num_term_in_collection = index_reader.stats()['total_terms']
    return np.log(num_term_in_collection / (index_reader.get_term_counts(term)[1]+1))

def bm25_term(docid: str, term: str, k1=0.82, b=0.68):
    return index_reader.compute_bm25_term_weight(docid, term)
def bm25_doc(docid):
    return {term: index_reader.compute_bm25_term_weight(docid, term, analyzer=None) for term in tf_doc(docid).keys()}

# static doc/passage features
def doc_len(docid: str):
    doc_vec = tf_doc(docid)
    doclen = 0
    for token in doc_vec:
        if type(doc_vec[token]) is int:
            doclen += doc_vec[token]
    return doclen

# query performance feature
def SCQ_d_t(docid: str, term:str):
    # similarity between query and collection
    f_c_t = index_reader.get_term_counts(term)[1]/index_reader.stats()["total_terms"]
    num_docs = index_reader.stats()['documents']
    f_t = index_reader.get_term_counts(term)[0]
    f_t = f_t if f_t != 0 else 0.5
    # print("f_c_t: ", f_c_t)
    # print("N/f_t: ", num_docs/f_t)
    return ( 1+np.log(f_c_t+1) ) * np.log(1+num_docs/f_t)

def VAR_t(term: str):
    # variability of query w.r.t collection
    # mean_w_t = index_reader.get
    postings_list = random.sample(index_reader.get_postings_list(term), 1)
    num_docs = index_reader.get_term_counts(term)[0]
    # docids = [p.docid for p in postings_list]
    w_t = []
    # for p in tqdm(postings_list, desc="traverse postings"):
    for p in postings_list:
        w_t += [1+np.log(p.tf) * np.log(1+index_reader.stats()['documents']/num_docs)]
    mean_w_t = sum(w_t) / len(postings_list)

    gamma1 = np.sqrt(sum(list(map(lambda x: np.square(x-mean_w_t), w_t))) / num_docs)

    return gamma1

# query features
def query_term_count(query: str):
    return len(index_reader.analyze(query))

# query-log based feature
def query_n_gram_score(query, docid):
    # print(type(searcher.doc(docid).raw()))
    doc = json.loads(searcher.doc(docid).raw())['contents']
    sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(doc)]
    # print(sentences)
    tokenized_query = nltk.word_tokenize(query)
    
    return sentence_bleu(sentences, tokenized_query, smoothing_function=SmoothingFunction().method4)

def query_expansion(query: str):
    # 
    local_searcher = SimpleSearcher(index_dir)
    set_bm25_parameters(local_searcher, index_dir, k1=0.82, b=0.68)
    hits = local_searcher.search(query, k=5)
    # docids = [hit.docid for hit in hits]
    passages = ""
    for hit in hits:
        passages += json.loads(local_searcher.doc(hit.docid).raw())['contents'] + " "
    
    return query+" "+passages

def calc_n_format_feats(qrel_dir:str=train_label_dir, query_dir=train_query_dir):
    '''
    return: x, y, proc_query_dict, proc_docid_list
    '''
    x = []
    y = []
    proc_query_dict = {}
    proc_docid_list = []
    group = []

    # build query retrieval list
    with open(query_dir) as fd:
        rd = csv.reader(fd, delimiter="\t")
        rd_list = [entry for entry in rd]

    # load & process training examples
    docid_list = get_docid_list_from_file(docid_list_dir)
    
    with open(qrel_dir) as fd:
        rd = csv.reader(fd, delimiter="\t")
        training_examples = [entry for entry in rd]
    training_examples = random.sample(training_examples, int(len(training_examples)/5))
    # print(len(training_examples))
    for row in tqdm(training_examples):
        qid, _, docid_base, _ = row
        query = [q_entry[1] for q_entry in rd_list if q_entry[0] == qid][0]
        tokenized_query = []

        for token in index_reader.analyze(query):
            tokenized_query += index_reader.analyze(token)
        # print(tokenized_query)
        
        # 1 more negative, total 2 training examples for 1 qrel entry
        docids_per_qrel = set(random.sample(docid_list, 1)+[docid_base])
        group.append(len(docids_per_qrel))
        
        for docid in docids_per_qrel:
            # extra feature 1: bleu score:
            bleu_score = query_n_gram_score(query, docid)
            
            proc_docid_list.append(docid)
            label = 1 if docid==docid_base else 0
            y.append(label)
            proc_query_dict[qid] = query

            # calculate local features 
            bm25_score, tfidf_score, tf_score, ictf_score = 0, 0, 0, 0
            scq_score, gamma1_score = [], []
            for token in tokenized_query:
                # print(SCQ_d_t(docid, token))
                try:
                    bm25_score += bm25_term(docid, token)
                    tfidf_score += tfidf_t_d(docid, token)
                    # extra feature 2: tf
                    tf_score += tf_t_d(docid, token)
                    # extra feature 3: ictf
                    ictf_score += ictf_t_d(docid, token)
                    scq_score.append(SCQ_d_t(docid, token))
                    # gamma1_score.append(VAR_t(token))
                except Exception:
                    # pass
                    print(qid)
            
            doclen = doc_len(docid)
            querylen = len(tokenized_query)
            # extra feature 3: SCQ_score
            try:
                max_scq_score = max(scq_score)
            except:
                print("qid: ", qid)
                print("docid: ", docid)
            # extra feature 4: gamma1_score
            # max_gamma1_score = max(gamma1_score)
            
            x.append([bm25_score, tfidf_score, doclen, querylen, bleu_score, tf_score, ictf_score, max_scq_score])
            # x.append([bm25_score, tfidf_score, doclen, querylen, tf_score, ictf_score, max_scq_score, max_gamma1_score])

    return x, y, proc_query_dict, proc_docid_list, group

def dump_feats(feats_dir=train_feats_dir, query_dir=train_label_dir, qid_dir=train_qids, docid_dir=train_docids, group_dir=group_dir):
    x, y, proc_query_dict, proc_docid_list, group = calc_n_format_feats(query_dir)
    
    with open(qid_dir, "w") as fp:
        json.dump(proc_query_dict, fp)
    with open(docid_dir, "w") as fp:
        json.dump(proc_docid_list, fp)
    with open(group_dir, "w") as fp:
        json.dump(group, fp)
    
    ds.dump_svmlight_file(csc_matrix(x), np.array(y), feats_dir)


if __name__ == "__main__":
    # print("enter main")
    # docid_list = get_docid_list(collection_name, collection_dir, docid_list_dir)
    # docid_list = get_docid_list_from_file(docid_list_dir)
    # test_docid = docid_list[0]

    # print(len(docid_list))
    # print(test_docid)
    
    # doc = searcher.doc(test_docid)
    # print(doc.raw())

    new_train_feats = "features/refined_features/refined_train_feats.txt"
    dump_feats(feats_dir=new_train_feats)
    
    # error analysis
    # queries = [
    #     "who is robert gray",
    #     "define visceral?",
    #     "when was the salvation army founded",
    #     "what is the daily life of thai people",
    #     "right pelvic pain causes",
    #     "what are the three percenters?"
    # ]
    # print(query_n_gram_score(queries[2], test_docid))
    # print(ictf_t_d(test_docid, "define"))
    # print(index_reader.get_term_counts("pelvis"))
    # print(type(index_reader.get_postings_list("pelvis")[0]))
    # print(SCQ_d_t("pelvis", test_docid))
    # print(VAR_t("pelvis"))
    

