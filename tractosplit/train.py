import os
import argparse
import datetime
import tractosplit.utils.constants as constants
from tractosplit.models.LSTM.lstm_classifier import lstmClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("classifier", type=str, help="Name of classifier to train")
    parser.add_argument(
        "-t", nargs="+", help="Training subjects folders", required=True
    )
    parser.add_argument(
        "-v", nargs="+", help="Validation subjects folders", required=True
    )
    args = parser.parse_args()

    print("[INFO] Training", args.classifier)
    print("[INFO] Training subjects:", args.t)
    print("[INFO] Validation subjects:", args.v)
    train_id = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "/"
    checkpoint_dir = constants.lstm_path + train_id
    report_dir = constants.train_report_path + train_id
    os.mkdir(checkpoint_dir)
    os.mkdir(report_dir)

    try:
        lstm = lstmClassifier()
        lstm.train(args.t, args.v, train_id)
    except Exception as e:
        os.rmdir(checkpoint_dir)
        os.rmdir(report_dir)
        raise e

