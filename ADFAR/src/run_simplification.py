import argparse
import csv
import pandas
from tqdm import tqdm
import os
import simplification


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--complex_threshold",
        type=float,
        default=0.1,
        help="Threshold is defined differently with different simplifiy_version.",
    )
    parser.add_argument(
        "--simplify_version", type=str, default="v1", help="Choose from v1 to v6."
    )
    parser.add_argument(
        "--file_to_simplify", type=str, help="Input path for original train.tsv."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for simplified train.tsv.",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.2, help="ratio of words to be changed."
    )
    parser.add_argument(
        "--syn_num", type=int, default=10, help="ratio of words to be changed."
    )
    parser.add_argument(
        "--most_freq_num", type=int, default=5, help="ratio of words to be changed."
    )
    parser.add_argument(
        "--cos_sim_file",
        type=str,
        default="cos_sim_counter_fitting/cos_sim_counter_fitting.npy",
        help="ratio of words to be changed.",
    )
    parser.add_argument(
        "--counterfitted_vectors",
        type=str,
        default="embeddings/subword_dnabert2_embeddings.txt",
        help="ratio of words to be changed.",
    )
    parser.add_argument(
        "--freq_file",
        type=str,
        default="dataset_frequencies.json",
        help="ratio of words to be changed.",
    )

    args = parser.parse_args()
    simplifier = simplification.Simplifier(
        threshold=args.complex_threshold,
        ratio=args.ratio,
        syn_num=args.syn_num,
        most_freq_num=args.most_freq_num,
        cos_sim_file=args.cos_sim_file,
        counterfitted_vectors=args.counterfitted_vectors,
        frequency_dict=args.freq_file,
    )
    simplify_dict = {
        "v2": simplifier.simplify_v2,
        "random_freq_v1": simplifier.random_freq_v1,
        "random_freq_v2": simplifier.random_freq_v2,
    }
    simplify = simplify_dict[args.simplify_version]
    # read data
    IDs = []
    sents = []
    labels = []
    with open(args.file_to_simplify, "r", encoding="utf-8-sig") as f:
        data = csv.reader(f, delimiter="\t")
        for row in data:
            if row[1] == "sentence":
                continue
            sents.append(row[0])
            labels.append(row[1])
    # simplify sentences
    simp_sents = []
    for sent in tqdm(sents):
        simp_sent = simplify(sent)
        simp_sents.append(simp_sent)

    # store simplified results
    simp_dataframe = pandas.DataFrame({"sentence": simp_sents, "label": labels})
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    simp_dataframe.to_csv(
        os.path.join(args.output_path, "train.tsv"), sep="\t", index=False
    )


if __name__ == "__main__":
    main()
