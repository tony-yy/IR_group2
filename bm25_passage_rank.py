import argparse
import os
from typing import Tuple, List, TextIO

from pyserini.pyclass import autoclass
from pyserini.search import get_topics, SimpleSearcher, JSimpleSearcherResult
from pyserini.search.reranker import ClassifierType, PseudoRelevanceClassifierReranker
from pyserini.query_iterator import QUERY_IDS, query_iterator
from tqdm import tqdm

import csv

def write_result(target_file: TextIO, result: Tuple[str, List[JSimpleSearcherResult]],
                 hits_num: int, msmarco: bool, tag: str):
    topic, hits = result
    docids = [hit.docid.strip() for hit in hits]
    docids_unstripped = [hit.docid for hit in hits]
    
    scores = [hit.score for hit in hits]

    if msmarco:
        for i, docid in enumerate(docids):
            if i >= hits_num:
                break
            target_file.write(f'{topic}\t{docid}\t{i + 1}\n')
    else:
        for i, (docid, score) in enumerate(zip(docids, scores)):
            if i >= hits_num:
                break
            target_file.write(
                f'{topic} Q0 {docid} {i + 1} {score:.6f} {tag}\n')


def set_bm25_parameters(searcher, index, k1=None, b=None):
    if k1 is not None or b is not None:
        if k1 is None or b is None:
            print('Must set *both* k1 and b for BM25!')
            exit()
        print(f'Setting BM25 parameters: k1={k1}, b={b}')
        searcher.set_bm25(k1, b)
    else:
        # Automatically set bm25 parameters based on known index:
        if index == 'msmarco-passage' or index == 'msmarco-passage-slim':
            print('MS MARCO passage: setting k1=0.82, b=0.68')
            searcher.set_bm25(0.82, 0.68)
        elif index == 'msmarco-passage-expanded':
            print('MS MARCO passage w/ doc2query-T5 expansion: setting k1=2.18, b=0.86')
            searcher.set_bm25(2.18, 0.86)
        elif index == 'msmarco-doc' or index == 'msmarco-doc-slim':
            print('MS MARCO doc: setting k1=4.46, b=0.82')
            searcher.set_bm25(4.46, 0.82)
        elif index == 'msmarco-doc-per-passage' or index == 'msmarco-doc-per-passage-slim':
            print('MS MARCO doc, per passage: setting k1=2.16, b=0.61')
            searcher.set_bm25(2.16, 0.61)
        elif index == 'msmarco-doc-expanded-per-doc':
            print('MS MARCO doc w/ doc2query-T5 (per doc) expansion: setting k1=4.68, b=0.87')
            searcher.set_bm25(4.68, 0.87)
        elif index == 'msmarco-doc-expanded-per-passage':
            print('MS MARCO doc w/ doc2query-T5 (per passage) expansion: setting k1=2.56, b=0.59')
            searcher.set_bm25(2.56, 0.59)


def define_search_args(parser):
    parser.add_argument('--index', type=str, metavar='path to index or index name', required=True,
                        help="Path to Lucene index or name of prebuilt index.")
    parser.add_argument('--bm25', action='store_true', default=True, help="Use BM25 (default).")
    parser.add_argument('--k1', type=float, help='BM25 k1 parameter.')
    parser.add_argument('--b', type=float, help='BM25 b parameter.')

    parser.add_argument('--rm3', action='store_true', help="Use RM3")
    parser.add_argument('--qld', action='store_true', help="Use QLD")

    parser.add_argument('--prcl', type=ClassifierType, nargs='+', default=[],
                        help='Specify the classifier PseudoRelevanceClassifierReranker uses.')
    parser.add_argument('--prcl.vectorizer', dest='vectorizer', type=str,
                        help='Type of vectorizer. Available: TfidfVectorizer, BM25Vectorizer.')
    parser.add_argument('--prcl.r', dest='r', type=int, default=10,
                        help='Number of positive labels in pseudo relevance feedback.')
    parser.add_argument('--prcl.n', dest='n', type=int, default=100,
                        help='Number of negative labels in pseudo relevance feedback.')
    parser.add_argument('--prcl.alpha', dest='alpha', type=float, default=0.5,
                        help='Alpha value for interpolation in pseudo relevance feedback.')


if __name__ == "__main__":
    # parameters
    index_path = "C:\\Users\\Tony\\anserini-pyserini\\pyserini\\indexes\\lucene-index-msmarco-passage"
    output_path = "C:\\Users\\Tony\\anserini-pyserini\\pyserini\\runs\\run.msmarco-passage-test.bm25tuned.txt"
    tag = '.txt'
    k1 = 0.82
    b = 0.68
    hits_num = 1000
    msmarco = True
    
    # set up searcher
    searcher = SimpleSearcher(index_path)
    set_bm25_parameters(searcher, index_path, k1, b)

    # load topics
    topic_file_path = "D:\\MSMARCO\\passage_retrieval\\raw\\msmarco-test2019-queries.tsv"
    topics = {}
    with open(topic_file_path) as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            topics[int(row[0])] = {'title': row[1]}

    with open(output_path, 'w') as fd:
        batch_topics = list()
        batch_topic_ids = list()
        for index, (topic_id, text) in enumerate(tqdm(list(query_iterator(topics, order=None)))):
            hits = searcher.search(text, hits_num)
            results = [(topic_id, hits)]
            print("docid: ", hits[0].docid)
            print(searcher.doc(hits[0].docid).contents)
            
            for result in results:
                write_result(fd, result, hits_num, msmarco, tag)
            results.clear()
