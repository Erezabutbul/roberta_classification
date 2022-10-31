import pandas as pd
from simpletransformers.classification import ClassificationModel
import sklearn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logger as logger
from pathlib import Path
import os


def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    confusion_matrix = str(sklearn.metrics.confusion_matrix(labels, preds)),
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': confusion_matrix
    }


def main():
    run_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    model = ClassificationModel('roberta', 'roberta-base', num_labels=6, use_cuda=False,
                                args={
                                    "output_dir": run_dir + r"\\",
                                    "reprocess_input_data": True, "overwrite_output_dir": True, "silent": True,
                                    'train_batch_size': 2, 'gradient_accumulation_steps': 16,
                                    'learning_rate': 3e-5, 'num_train_epochs': 3, 'max_seq_length': 512})

    start = 1
    for idx in range(start, 5, 1):
        file_path = os.path.join(run_dir, 'res', str(idx) + ".txt")
        file_location = Path(file_path)
        if file_location.exists():
            logger.print_msg("found :" + str(file_path))
            continue

        logger.print_msg("at idx: " + str(idx))

        train_str = "train_df_" + str(idx) + ".csv"
        eval_str = "eval_df_" + str(idx) + ".csv"
        train_df = pd.read_csv(os.path.join(run_dir, 'data', 'splits', train_str))
        model.train_model(train_df)

        eval_df = pd.read_csv(os.path.join(run_dir, 'data', 'splits', eval_str))
        result, model_outputs, wrong_predictions = model.eval_model(eval_df, compute_metrics=compute_metrics)
        with open(file_path, "w") as outfile:
            outfile.write(str(result))
        print(result)


if __name__ == '__main__':
    main()
