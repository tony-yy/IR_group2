if __name__ == '__main__':
    mrr_score_path = "mrr_per_query.txt"
    ndcg_cut_10_path = "ndcg_cut_10_query.txt"

    rank1 = []
    with open(mrr_score_path, "r") as fd:
        lines = fd.readlines()
    for line in lines:
        # print(line.split(" "))
        rank1.append(float(line.split(" ")[-1][:-1]))
    
    rank2 = []
    with open(ndcg_cut_10_path, "r") as fd:
        lines = fd.readlines()
        # print(lines)
    for line in lines:
        rank2.append(float(line.split(" ")[-1][:-1]))

    print(sorted(rank1))
    print(sorted(rank2))