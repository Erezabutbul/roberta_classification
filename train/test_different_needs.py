import pandas as pd
import sklearn.metrics
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging as logger
import os

run_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    confusion_matrix = str(sklearn.metrics.confusion_matrix(labels, preds))
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': confusion_matrix
    }


input_csv = os.path.join(run_dir, 'basic_test_folder', "basic_test.csv")
output_csv = os.path.join(run_dir, 'basic_test_folder', "basic_test_result.csv")
result_txt = os.path.join(run_dir, 'basic_test_folder', "basic_test_result.txt")



def main():
    list_res = list()
    model = ClassificationModel('roberta', run_dir, use_cuda=False)

    test_df = pd.read_csv(input_csv)
    labels = test_df["labels"].tolist()
    to_predict = test_df.text.apply(lambda x: x.replace('\n', ' ')).tolist()
    preds, outputs = model.predict(to_predict)
    for idx in range(len(to_predict)):
        logger.debug("at idx: " + str(idx))
        tmp_dict = dict()
        tmp_dict["labels"] = labels[idx]
        tmp_dict["pred"] = preds[idx]
        tmp_dict["other"] = outputs[idx][0]
        tmp_dict["apology"] = outputs[idx][1]
        tmp_dict["request"] = outputs[idx][2]
        tmp_dict["greetings"] = outputs[idx][3]
        tmp_dict["complaints"] = outputs[idx][4]
        tmp_dict["compliments"] = outputs[idx][5]
        tmp_dict["text"] = to_predict[idx]
        list_res.append(tmp_dict)

    df = pd.DataFrame(list_res)
    df = df.round(decimals=3)
    df.to_csv(output_csv, index=True, index_label="index", encoding="utf_8_sig")
    result = compute_metrics(df["labels"], df["pred"])
    print(result['confusion_matrix'])
    with open(result_txt, "w") as outfile:
        outfile.write(str(result))



if __name__ == '__main__':
    main()