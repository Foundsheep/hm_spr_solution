from pathlib import Path
import random
import numpy as np
from configs import Config

def seperate_data(root, train_ratio, val_ratio, test_ratio, shuffle=True):
    failure = Config.FAILURE
    success = Config.SUCCESS

    # check the ratio
    sum_ratio = train_ratio + val_ratio + test_ratio
    if str(sum_ratio) != "1.0" and str(sum_ratio) != "1":
        return failure
    
    # get the number of all files
    root_folder = Path(root)
    annotated_folder = root_folder / "annotated"
    raw_folder = root_folder / "raw"

    if not root_folder.exists() or not annotated_folder.exists() or not raw_folder.exists():
        print("necessary folders don't exist")
        return failure
    
    raw_files = sorted(list(raw_folder.glob("*")))
    num_files = len(raw_files)
    indices = np.arange(num_files)

    # get the numbers for each dataset
    num_train = round(num_files * train_ratio)
    num_val = round(num_files * val_ratio)
    num_test = num_files - num_train - num_val
    print(f"numbers of file to make: train=[{num_train}], val=[{num_val}], test=[{num_test}]")

    # check the sanity
    if train_ratio > 0 and num_train < 1:
        print(f"ratio is greater than 0[{train_ratio = }],  but that fraction didn't make any number to select in files[{num_train = }]")
        return failure
    elif val_ratio > 0 and num_val < 1:
        print(f"ratio is greater than 0[{val_ratio = }],  but that fraction didn't make any number to select in files[{num_val = }]")
        return failure
    elif test_ratio > 0 and num_test < 1:
        print(f"ratio is greater than 0[{test_ratio = }],  but that fraction didn't make any number to select in files[{num_test = }]")
        return failure

    # prepare the directories for copies
    train_folder = root_folder / "train"
    val_folder = root_folder / "val"
    test_folder = root_folder / "test"

    # if seperated before
    if train_folder.exists():
        print("seperated data exist already!")
        return success

    annotated_files = sorted(list(annotated_folder.glob("*.png")) + list(annotated_folder.glob("*.jpg")))
    labelmap_txt = list(annotated_folder.glob("labelmap.txt"))[0]

    # make commonly necessary folders
    if num_train > 0:
        train_folder.mkdir()
        (train_folder / "annotated").mkdir()
        (train_folder / "raw").mkdir()

    if num_val > 0:
        val_folder.mkdir()
        (val_folder / "annotated").mkdir()
        (val_folder / "raw").mkdir()

    if num_test > 0:
        test_folder.mkdir()
        (test_folder / "annotated").mkdir()
        (test_folder / "raw").mkdir()

    # shuffle
    if shuffle:
        random.shuffle(indices)
    
    # split data
    indices_train = indices[:num_train]
    indices_val = indices[num_train:num_train+num_val]
    indices_test = indices[num_train+num_val:num_train+num_val+num_test]

    # --- train
    count = 0
    for index in indices_train:
        path_raw = raw_files[index]
        path_annotated = annotated_files[index]

        copy_path_raw = train_folder / "raw" / path_raw.name
        copy_path_annotated = train_folder / "annotated" / path_annotated.name
        copy_path_labelmap_txt = train_folder / "annotated" / labelmap_txt.name

        copy_path_raw.write_bytes(path_raw.read_bytes())
        copy_path_annotated.write_bytes(path_annotated.read_bytes())
        copy_path_labelmap_txt.write_bytes(labelmap_txt.read_bytes())

        count += 1
    print(f"train saved [{count}] files, {num_train = }")

    # --- val
    count = 0
    for index in indices_val:
        path_raw = raw_files[index]
        path_annotated = annotated_files[index]

        copy_path_raw = val_folder / "raw" / path_raw.name
        copy_path_annotated = val_folder / "annotated" / path_annotated.name
        copy_path_labelmap_txt = val_folder / "annotated" / labelmap_txt.name

        copy_path_raw.write_bytes(path_raw.read_bytes())
        copy_path_annotated.write_bytes(path_annotated.read_bytes())
        copy_path_labelmap_txt.write_bytes(labelmap_txt.read_bytes())

        count += 1
    print(f"val saved [{count}] files, {num_val = }")

    # --- test
    count = 0
    for index in indices_test:
        path_raw = raw_files[index]
        path_annotated = annotated_files[index]

        copy_path_raw = test_folder / "raw" / path_raw.name
        copy_path_annotated = test_folder / "annotated" / path_annotated.name
        copy_path_labelmap_txt = test_folder / "annotated" / labelmap_txt.name

        copy_path_raw.write_bytes(path_raw.read_bytes())
        copy_path_annotated.write_bytes(path_annotated.read_bytes())
        copy_path_labelmap_txt.write_bytes(labelmap_txt.read_bytes())

        count += 1
    print(f"test saved [{count}] files, {num_test = }")
    return success