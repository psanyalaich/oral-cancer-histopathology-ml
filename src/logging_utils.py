import os
import csv
import logging
import traceback


def setup_experiment_logger(
    experiment_name,
    results_dir
):

    logger = logging.getLogger(
        experiment_name
    )

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    log_path = os.path.join(
        results_dir,
        "experiment.log"
    )

    file_handler = logging.FileHandler(
        log_path
    )

    formatter = logging.Formatter(
        "%(asctime)s - "
        "%(levelname)s - "
        "%(message)s"
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(file_handler)
    return logger


def log_experiment_failure(
    csv_path,
    experiment_name,
    fold,
    exception
):

    file_exists = os.path.exists(csv_path)

    with open(
        csv_path,
        "a",
        newline = "",
        encoding = "utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames = [
                "experiment",
                "fold",
                "exception_type",
                "exception_message",
                "traceback"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "experiment": experiment_name,
            "fold": fold,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc()
        })
