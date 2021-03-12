import os
import lightgbm as lgb
from sklearn import datasets as ds
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from sklearn.preprocessing import OneHotEncoder
# from data_format_read import read_dataset
# from ndcg import validate
import shap
import matplotlib.pyplot as plt
from extract_features import get_docid_list_from_file 
# import graphviz
import json
import csv
from tqdm import tqdm
from pyserini.search import SimpleSearcher
from pyserini.index import IndexReader
from extract_features import SCQ_d_t, bm25_term, doc_len, ictf_t_d, index_dir, query_expansion, query_n_gram_score, tf_t_d, tfidf_t_d, train_docids
import random
from scipy.sparse.csc import csc_matrix

searcher = SimpleSearcher(index_dir)
index_reader = IndexReader(index_dir)

def load_data(feats: str, group: str, specified_feats:list=None):
    '''
    load feature,label,query
    :param feats:
    :param group:
    :return:
    '''

    x_train, y_train = ds.load_svmlight_file(feats)
    # q_train = np.loadtxt(group)
    with open(group, "r") as fp:
        group = json.load(fp)
    assert sum(group) == y_train.shape[0] and y_train.shape[0] == x_train.shape[0]
    if specified_feats:
        x_train = csc_matrix(x_train.todense()[:, specified_feats])

    return x_train, y_train, group

def load_test_data_from_file(test_query_path: str, collection_docid_path: str):
    x = []

    # load & process training examples
    docid_list = list(set(get_docid_list_from_file(collection_docid_path)))
    docid_list = random.sample(docid_list, 50000)

    with open(test_query_path) as fd:
        rd = csv.reader(fd, delimiter="\t")
        test_entries = [entry for entry in rd][:1]
    
    for row in tqdm(test_entries):
        qid, query = row
        query = query_expansion(query)
        tokenized_query = []
        for token in index_reader.analyze(query):
            tokenized_query += index_reader.analyze(token)
        # print(tokenized_query)
        '''
        for docid in tqdm(docid_list):
            # calculate local features 
            bm25_score, tfidf_score = 0, 0
            for token in tokenized_query:
                # print(token)
                try:
                    bm25_score += bm25_term(docid, token)
                    tfidf_score += tfidf_t_d(docid, token)
                except Exception:
                    # pass
                    print(qid)
            doclen = doc_len(docid)
            querylen = len(tokenized_query)
            x.append([bm25_score, tfidf_score, doclen, querylen])
        '''
        bm25_score, tfidf_score, tf_score, ictf_score = 0, 0, 0, 0
        scq_score, gamma1_score = [], []
        for docid in docid_list:
            # extra feature 1: bleu score:
            bleu_score = query_n_gram_score(query, docid)
            
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
                except Exception:
                    # pass
                    print(qid) 
            doclen = doc_len(docid)
            querylen = len(tokenized_query)

            # extra feature 4: maxSCQ_score
            max_scq_score = max(scq_score)
            
            x.append([bm25_score, tfidf_score, doclen, querylen, bleu_score, tf_score, ictf_score, max_scq_score])

    return x, docid_list

def load_test_data_by_query(query: str, docid: str, feats_selection:list=None):
    query = query_expansion(query)
    x = []
    tokenized_query = []
    for token in index_reader.analyze(query):
        tokenized_query += index_reader.analyze(token)
    # print(tokenized_query)

    # calculate local features 
        bm25_score, tfidf_score, tf_score, ictf_score = 0, 0, 0, 0
        scq_score, gamma1_score = [], []

        # extra feature 1: bleu score:
        bleu_score = query_n_gram_score(query, docid)
        
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
            except Exception:
                # pass
                print(qid) 
        doclen = doc_len(docid)
        querylen = len(tokenized_query)

        # extra feature 4: maxSCQ_score
        max_scq_score = max(scq_score)
        
    feat_val = [bm25_score, tfidf_score, doclen, querylen, bleu_score, tf_score, ictf_score, max_scq_score]
    if feats_selection:
        feat_val = [feat_val[i] for i in feats_selection]
        
    x.append(feat_val)

    return x, docid

def train(x_train, y_train, q_train, model_save_path):
    train_data = lgb.Dataset(x_train, label=y_train, group=q_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbrt',
        'objective': 'lambdarank', 
        'metric': 'ndcg', 
        'max_position': 10,  
        'metric_freq': 1,  
        'train_metric': True,
        'ndcg_at': [10],
        'max_bin': 255,  
        'num_iterations': 250,
        'learning_rate': 0.01,  
        'num_leaves': 31, 
        'min_data_in_leaf': 30,  
        'verbose': 2
    }
    gbm = lgb.train(params, train_data, valid_sets=[train_data])
    gbm.save_model(model_save_path)

def predict(x_test, retrieved_docs: list, model_input_path, retri_docids:bool=False):

    gbm = lgb.Booster(model_file=model_input_path)  

    ypred = gbm.predict(x_test)
    # print(ypred)
    predicted_sorted_indexes = np.argsort(ypred)[::-1]  # return pred_val by reversed order
    # print(predicted_sorted_indexes)
    try:
        assert len(retrieved_docs) == len(predicted_sorted_indexes)
    except:
        print("original group docids: ", len(retrieved_docs), end="\n")
        print("docid ranking: ", len(predicted_sorted_indexes), end="\n")
        print(predicted_sorted_indexes)
    t_results = [retrieved_docs[i] for i in predicted_sorted_indexes] # return docid according to above index
    rst = t_results if retri_docids else (ypred, predicted_sorted_indexes)
    return rst

def plot_print_feature_shap(model_path, data_feats, feats_col_name):

    gbm = lgb.Booster(model_file=model_path)
    gbm.params["objective"] = "regression"
    
    X_train, _ = ds.load_svmlight_file(data_feats)
    # print("original feature shape:\n", X_train, end="\n")
    #features
    feature_mat = X_train.todense()
    df_feature = pd.DataFrame(feature_mat)
    # print("dataframed feature shape", df_feature)

    df_feature.columns = feats_col_name
    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(df_feature[feats_col_name])

    shap.summary_plot(shap_values, df_feature[feats_col_name], plot_type="bar")
    shap.summary_plot(shap_values, df_feature[feats_col_name])

    # impact of interaction of two features on the result
    feat_name1, feat_name2 = feats_col_name[0], feats_col_name[1]
    shap.dependence_plot(feat_name1, shap_values, df_feature[feats_col_name], interaction_index=feat_name2, show=True)



if __name__ == '__main__':

    data_feats = "features/baseline_features/train_feats.txt"
    data_group = "features/baseline_features/group.txt"
    save_plot_path = "anlyz/plot/tree_plot"

    baseline_model_path = "model/model_baseline.md"
    # refined_model_path = "model/model_refined.md"

    test_query_path = "D:/MSMARCO/passage_retrieval/raw/msmarco-test2019-queries.tsv"
    new_data_feats = "features/refined_features/refined_train_feats.txt"
    new_data_group = "features/group.txt"
    # feats_selection = {"sel1":[0,1,2,4], "sel2":[0,1,2,5],"sel3":[0,1,2,6],
    #                         "sel4": [0,1,2,7],"sel5":[0,1,2,4,5],"sel6": [0,1,2,4,6],
    #                         "sel7":[0,1,2,4,7]
    #                 }
    feats_selection = {"sel1":[0,1,2,7]}

    if sys.argv[1] == '-train':
        # per feature analysis
        for sel in feats_selection:
            crnt_model_path = "model/model_refined_"+sel+".md"
            x_train, y_train, q_train = load_data(new_data_feats, new_data_group, feats_selection[sel])
            train(x_train, y_train, q_train, crnt_model_path)
    
    elif sys.argv[1] == '-test':
        # test
        # result_path = "results/baseline/l2r_baseline_rst.trec"
        # new_result_path = "results/improved/l2r_refined_rst.trec"
        test_qrel_path = "D:/MSMARCO/passage_retrieval/raw/test_data/2019qrels-pass.txt"
        test_query_path = "D:/MSMARCO/passage_retrieval/raw/test_data/msmarco-test2019-queries.tsv"

        with open(test_query_path) as fd:
            rd = csv.reader(fd, delimiter="\t")
            query_list = [line for line in rd]
        
        with open(test_qrel_path, "r") as fp:
            rd = csv.reader(fp, delimiter="\t")
            test_line = [line for line in rd]
        
        # group test_line with qid
        grouped_test_entry = {}
        for line in tqdm(test_line, desc="group test entries with qid"):
            qid, _, docid, _ = line[0].split(" ")
            if qid in grouped_test_entry:
                grouped_test_entry[qid] += [docid]
            else:
                grouped_test_entry[qid] = [docid]

        rst_to_dump = ""
        
        for sel in feats_selection:
            crnt_rst_path = "results/improved/l2r_refined_rst_"+sel+".trec"
            crnt_model_path = "model/model_refined_"+sel+".md"
            for qid in tqdm(grouped_test_entry, desc="start predicting"):
                test_x = []
                # print(grouped_test_entry[qid])
                for docid in grouped_test_entry[qid]:
                    query = [q_entry[1] for q_entry in query_list if q_entry[0] == qid][0]
                    x, _ = load_test_data_by_query(query, docid, feats_selection[sel])
                    # print(x)
                    test_x += x

                # print(len(test_x))
                # assert False
                # print(len(grouped_test_entry[qid]))
                score, rank_index = predict(test_x, grouped_test_entry[qid], crnt_model_path)
                assert len(score.tolist()) == len(rank_index.tolist())
                for s, r, d in zip(score.tolist(), rank_index.tolist(), grouped_test_entry[qid]):
                    rst_to_dump += qid+" "+"Q0"+" "+d+" "+str(r+1)+" "+str(s)+" "+"STANDARD"+"\n"

            with open(crnt_rst_path, "w") as fp:
                fp.write(rst_to_dump)
    
    elif sys.argv[1] == '-shap':
        mode_path_1 = "model/model_refined_sel1.md"
        mode_path_2 = "model/model_refined_sel2.md"
        mode_path_3 = "model/model_refined_sel3.md"
        mode_path_4 = "model/model_refined_sel4.md"
        mode_path_5 = "model/model_refined_sel7.md"

        feats_col_name1 = ['bm25', 'tf_idf', 'passageLength', 'bleu_score']
        feats_col_name2 = ['bm25', 'tf_idf', 'passageLength', 'tf_score']
        feats_col_name3 = ['bm25', 'tf_idf', 'passageLength', 'ictf_score']
        feats_col_name4 = ['bm25', 'tf_idf', 'passageLength', 'max_scq_score']
        feats_col_name5 = ['bm25', 'tf_idf', 'passageLength', 'bleu_score', 'tf_score']

        # plot_print_feature_shap(mode_path_1, data_feats, 1, feats_col_name1)
        # plot_print_feature_shap(mode_path_2, data_feats, 1, feats_col_name2)
        # plot_print_feature_shap(mode_path_3, data_feats, 1, feats_col_name3)
        # plot_print_feature_shap(mode_path_4, data_feats, 1, feats_col_name4)
        plot_print_feature_shap(mode_path_5, data_feats, feats_col_name5)

    