import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from weakly_supervised_parser.settings import PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
from weakly_supervised_parser.settings import PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH
from weakly_supervised_parser.utils.prepare_dataset import DataLoaderHelper
from weakly_supervised_parser.inference import process_test_sample
from weakly_supervised_parser.tree.helpers import get_constituents, get_distituents


def prepare_data_for_self_training(
    inside_model, train_initial, valid_initial, threshold, num_train_rows, num_valid_examples, seed, scale_axis, predict_batch_size
):
    train_sentences = DataLoaderHelper(input_file_object=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
    train_gold_file_path = PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
    lst = []
    for train_index, train_sentence in enumerate(train_sentences):

        if train_index == num_train_rows:
            break

        best_parse = process_test_sample(
            train_index,
            train_sentence,
            train_gold_file_path,
            predict_type="inside",
            scale_axis=scale_axis,
            predict_batch_size=predict_batch_size,
            model=inside_model,
        )

        best_parse_get_constituents = get_constituents(best_parse)
        best_parse_get_distituents = get_distituents(best_parse)

        if best_parse_get_constituents:
            constituents_proba = inside_model.predict_proba(
                pd.DataFrame(dict(sentence=best_parse_get_constituents)), scale_axis=scale_axis, predict_batch_size=predict_batch_size
            )[:, 1]
            df_constituents = pd.DataFrame({"sentence": best_parse_get_constituents, "label": constituents_proba})
            df_constituents["label"] = np.where(df_constituents["label"] > threshold, 1, -1)

        if best_parse_get_distituents:
            distituents_proba = inside_model.predict_proba(pd.DataFrame(dict(sentence=best_parse_get_distituents)))[:, 0]
            df_distituents = pd.DataFrame({"sentence": best_parse_get_distituents, "label": distituents_proba})
            df_distituents["label"] = np.where(df_distituents["label"] > threshold, 0, -1)

        if best_parse_get_constituents and best_parse_get_distituents:
            out = pd.concat([df_constituents, df_distituents])

        elif best_parse_get_constituents and not best_parse_get_distituents:
            out = df_constituents

        elif best_parse_get_distituents and not best_parse_get_constituents:
            out = df_distituents

        lst.append(out)

    df_out = pd.concat(lst).sample(frac=1.0, random_state=seed)
    df_out.drop_duplicates(subset=["sentence"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    valid_idx = np.concatenate(
        (
            df_out[df_out["label"] == 1].index.values[: int(num_valid_examples // 4)],
            df_out[df_out["label"] == 0].index.values[: int(num_valid_examples // (4 / 3))],
        )
    )
    valid_df = df_out.loc[valid_idx]
    train_idx = df_out.loc[~df_out.index.isin(valid_idx)].index.values
    train_df = df_out.loc[np.concatenate((train_idx, df_out[df_out["label"] == -1].index.values))]

    train_augmented = pd.concat([train_initial, train_df]).drop_duplicates(subset=["sentence"])
    valid_augmented = pd.concat([valid_initial, valid_df]).drop_duplicates(subset=["sentence"])

    return train_augmented, valid_augmented


def prepare_outside_strings(inside_model, upper_threshold, lower_threshold, num_train_rows, seed, scale_axis, predict_batch_size):
    train_sentences = DataLoaderHelper(input_file_object=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
    train_gold_file_path = PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
    lst = []
    for train_index, train_sentence in enumerate(train_sentences):

        if train_index == num_train_rows:
            break

        best_parse, df = process_test_sample(
            train_index,
            train_sentence,
            train_gold_file_path,
            predict_type="inside",
            model=inside_model,
            scale_axis=scale_axis,
            predict_batch_size=predict_batch_size,
            return_df=True,
        )

        outside_constituent_samples = pd.DataFrame(dict(sentence=df.loc[df["scores"] > upper_threshold, "outside_sentence"].values, label=1))

        outside_distituent_samples = pd.DataFrame(dict(sentence=df.loc[df["scores"] < lower_threshold, "outside_sentence"].values, label=0))

        lst.append(pd.concat([outside_constituent_samples, outside_distituent_samples]))

    df_outside = pd.concat(lst, ignore_index=True)
    df_outside.drop_duplicates(subset=["sentence"], inplace=True)
    df_outside.reset_index(drop=True, inplace=True)
    train_outside, validation_outside = train_test_split(df_outside, test_size=0.5, random_state=seed, shuffle=True)
    return train_outside, validation_outside


def prepare_data_for_co_training(inside_model, outside_model, upper_threshold, lower_threshold, seed, scale_axis, predict_batch_size):
    train_sentences = DataLoaderHelper(input_file_object=PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH).read_lines()
    train_gold_file_path = PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH

    for train_index, train_sentence in enumerate(train_sentences):
        _, df_inside = process_test_sample(
            train_index,
            train_sentence,
            train_gold_file_path,
            predict_type="inside",
            model=inside_model,
            scale_axis=scale_axis,
            predict_batch_size=predict_batch_size,
            return_df=True,
        )
        _, df_outside = process_test_sample(
            train_index,
            train_sentence,
            train_gold_file_path,
            predict_type="outside",
            model=outside_model,
            scale_axis=scale_axis,
            predict_batch_size=predict_batch_size,
            return_df=True,
        )

        print(df_outside["outside_scores"].describe())

        outside_constituent_from_most_confident_inside = df_inside[df_inside["scores"] > upper_threshold]
        outside_constituent_from_most_confident_inside["label"] = 1
        outside_distituent_from_most_confident_inside = df_inside[df_inside["scores"] < lower_threshold]
        outside_distituent_from_most_confident_inside["label"] = 0
        outside_from_least_confident_inside = df_inside[(df_inside["scores"] > lower_threshold + 0.3) & (df_inside["scores"] < upper_threshold - 0.3)]
        outside_from_least_confident_inside["label"] = -1

        outside_from_confident_inside = pd.concat(
            [outside_constituent_from_most_confident_inside, outside_distituent_from_most_confident_inside, outside_from_least_confident_inside],
            ignore_index=True,
        )

        inside_constituent_from_most_confident_outside = df_outside[df_outside["scores"] > upper_threshold]
        inside_constituent_from_most_confident_outside["label"] = 1
        inside_distituent_from_most_confident_outside = df_outside[df_outside["scores"] < lower_threshold]
        inside_distituent_from_most_confident_outside["label"] = 0
        inside_from_least_confident_outside = df_outside[
            (df_outside["scores"] > lower_threshold + 0.3) & (df_outside["scores"] < upper_threshold - 0.3)
        ]
        inside_from_least_confident_outside["label"] = -1

        inside_from_confident_outside = pd.concat(
            [inside_constituent_from_most_confident_outside, inside_distituent_from_most_confident_outside, inside_from_least_confident_outside],
            ignore_index=True,
        )

        df_out = pd.concat([inside_from_confident_outside, outside_from_confident_inside]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(df_out["label"].value_counts())
        print(df_out["outside_scores"].describe())
        print(df_out["inside_scores"].describe())
        inside_strings = df_out["inside_sentence"].values
        outside_strings = df_out["outside_sentence"].values
        labels = df_out["label"].values

        return inside_strings, outside_strings, labels
