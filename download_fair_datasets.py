import argparse
import io
import os
import tempfile
import zipfile
from urllib import request

import numpy as np
import pandas as pd


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_file(url, save_path):
    print(f"Downloading: {url}")
    request.urlretrieve(url, save_path)
    print(f"Saved: {save_path}")


def build_compas(data_root):
    save_dir = os.path.join(data_root, "compas")
    ensure_dir(save_dir)

    raw_path = os.path.join(save_dir, "compas-scores-two-years.csv")
    if not os.path.exists(raw_path):
        download_file(
            "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
            raw_path,
        )

    df = pd.read_csv(raw_path)
    keep_cols = [
        "age",
        "c_charge_degree",
        "race",
        "age_cat",
        "score_text",
        "sex",
        "priors_count",
        "days_b_screening_arrest",
        "decile_score",
        "is_recid",
        "two_year_recid",
    ]
    df = df[
        (df["days_b_screening_arrest"] >= -30)
        & (df["days_b_screening_arrest"] <= 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ]
    df = df[df["race"].isin(["African-American", "Caucasian"])]
    df = df[keep_cols].dropna().copy()

    save_path = os.path.join(save_dir, "compas.csv")
    df.to_csv(save_path, index=False)
    print(f"[compas] shape={df.shape}, saved={save_path}")


def build_bank(data_root):
    save_dir = os.path.join(data_root, "bank")
    ensure_dir(save_dir)

    zip_path = os.path.join(save_dir, "bank.zip")
    if not os.path.exists(zip_path):
        download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
            zip_path,
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(save_dir)

    raw_path = os.path.join(save_dir, "bank-full.csv")
    df = pd.read_csv(raw_path, sep=";")
    df["age_group"] = np.where((df["age"] < 25) | (df["age"] > 60), "1", "0")
    df["y"] = df["y"].map({"yes": 1, "no": 0}).astype(int)

    cols = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "age_group",
        "y",
    ]
    df = df[cols].dropna().copy()
    save_path = os.path.join(save_dir, "bank.csv")
    df.to_csv(save_path, index=False)
    print(f"[bank] shape={df.shape}, saved={save_path}")


def build_lsac(data_root):
    save_dir = os.path.join(data_root, "lsac")
    ensure_dir(save_dir)

    raw_path = os.path.join(save_dir, "law_school_clean.csv")
    if not os.path.exists(raw_path):
        download_file(
            "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/law_school_clean.csv",
            raw_path,
        )

    df = pd.read_csv(raw_path).dropna().copy()
    df["race_bin"] = np.where(df["race"].astype(str).str.lower() == "white", "White", "Non-White")
    df["pass_bar"] = df["pass_bar"].astype(int)
    df["male"] = df["male"].astype(int).astype(str)

    cols = [
        "decile1b",
        "decile3",
        "lsat",
        "ugpa",
        "zfygpa",
        "zgpa",
        "fulltime",
        "fam_inc",
        "male",
        "tier",
        "race_bin",
        "pass_bar",
    ]
    df = df[cols].dropna().copy()
    save_path = os.path.join(save_dir, "lsac.csv")
    df.to_csv(save_path, index=False)
    print(f"[lsac] shape={df.shape}, saved={save_path}")


def convert_h181_ssp_to_csv(ssp_zip_path, h181_csv_path):
    with zipfile.ZipFile(ssp_zip_path, "r") as zf:
        ssp_names = [member for member in zf.namelist() if member.lower().endswith(".ssp")]
        if not ssp_names:
            raise RuntimeError(f"No .ssp file found in {ssp_zip_path}")
        with zf.open(ssp_names[0]) as f:
            # h181.ssp is SAS XPORT format.
            df = pd.read_sas(f, format="xport")

    df.to_csv(h181_csv_path, index=False)
    print(f"Converted {ssp_zip_path} -> {h181_csv_path} with shape={df.shape}")


def preprocess_meps_panel19(h181_csv_path, meps_csv_path):
    df = pd.read_csv(h181_csv_path)

    def race_fn(row):
        if (row["HISPANX"] == 2) and (row["RACEV2X"] == 1):
            return "White"
        return "Non-White"

    df["RACEV2X"] = df.apply(lambda row: race_fn(row), axis=1)
    df = df.rename(columns={"RACEV2X": "RACE"})
    df = df[df["PANEL"] == 19]
    df = df.rename(
        columns={
            "FTSTU53X": "FTSTU",
            "ACTDTY53": "ACTDTY",
            "HONRDC53": "HONRDC",
            "RTHLTH53": "RTHLTH",
            "MNHLTH53": "MNHLTH",
            "CHBRON53": "CHBRON",
            "JTPAIN53": "JTPAIN",
            "PREGNT53": "PREGNT",
            "WLKLIM53": "WLKLIM",
            "ACTLIM53": "ACTLIM",
            "SOCLIM53": "SOCLIM",
            "COGLIM53": "COGLIM",
            "EMPST53": "EMPST",
            "REGION53": "REGION",
            "MARRY53X": "MARRY",
            "AGE53X": "AGE",
            "POVCAT15": "POVCAT",
            "INSCOV15": "INSCOV",
        }
    )

    df = df[df["REGION"] >= 0]
    df = df[df["AGE"] >= 0]
    df = df[df["MARRY"] >= 0]
    df = df[df["ASTHDX"] >= 0]

    bounded_cols = [
        "FTSTU",
        "ACTDTY",
        "HONRDC",
        "RTHLTH",
        "MNHLTH",
        "HIBPDX",
        "CHDDX",
        "ANGIDX",
        "EDUCYR",
        "HIDEG",
        "MIDX",
        "OHRTDX",
        "STRKDX",
        "EMPHDX",
        "CHBRON",
        "CHOLDX",
        "CANCERDX",
        "DIABDX",
        "JTPAIN",
        "ARTHDX",
        "ARTHTYPE",
        "ASTHDX",
        "ADHDADDX",
        "PREGNT",
        "WLKLIM",
        "ACTLIM",
        "SOCLIM",
        "COGLIM",
        "DFHEAR42",
        "DFSEE42",
        "ADSMOK42",
        "PHQ242",
        "EMPST",
        "POVCAT",
        "INSCOV",
    ]
    df = df[(df[bounded_cols] >= -1).all(1)]

    util = df["OBTOTV15"] + df["OPTOTV15"] + df["ERTOT15"] + df["IPNGTD15"] + df["HHTOTD15"]
    df["UTILIZATION"] = np.where(util >= 10.0, 1, 0)

    cols = [
        "REGION",
        "AGE",
        "SEX",
        "RACE",
        "MARRY",
        "FTSTU",
        "ACTDTY",
        "HONRDC",
        "RTHLTH",
        "MNHLTH",
        "HIBPDX",
        "CHDDX",
        "ANGIDX",
        "MIDX",
        "OHRTDX",
        "STRKDX",
        "EMPHDX",
        "CHBRON",
        "CHOLDX",
        "CANCERDX",
        "DIABDX",
        "JTPAIN",
        "ARTHDX",
        "ARTHTYPE",
        "ASTHDX",
        "ADHDADDX",
        "PREGNT",
        "WLKLIM",
        "ACTLIM",
        "SOCLIM",
        "COGLIM",
        "DFHEAR42",
        "DFSEE42",
        "ADSMOK42",
        "PCS42",
        "MCS42",
        "K6SUM42",
        "PHQ242",
        "EMPST",
        "POVCAT",
        "INSCOV",
        "PERWT15F",
        "UTILIZATION",
    ]
    df = df[cols].dropna().copy()
    df.to_csv(meps_csv_path, index=False)
    print(f"[meps] shape={df.shape}, saved={meps_csv_path}")


def build_meps(data_root):
    save_dir = os.path.join(data_root, "meps")
    ensure_dir(save_dir)

    ssp_zip_path = os.path.join(save_dir, "h181ssp.zip")
    h181_csv_path = os.path.join(save_dir, "h181.csv")
    meps_csv_path = os.path.join(save_dir, "meps.csv")

    if not os.path.exists(ssp_zip_path):
        download_file("https://meps.ahrq.gov/mepsweb/data_files/pufs/h181ssp.zip", ssp_zip_path)

    if not os.path.exists(h181_csv_path):
        convert_h181_ssp_to_csv(ssp_zip_path, h181_csv_path)

    preprocess_meps_panel19(h181_csv_path, meps_csv_path)


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess fairness-focused datasets")
    parser.add_argument(
        "--dataname",
        type=str,
        default="all",
        choices=["all", "compas", "bank", "lsac", "meps"],
        help="Dataset to build",
    )
    args = parser.parse_args()

    builders = {
        "compas": build_compas,
        "bank": build_bank,
        "lsac": build_lsac,
        "meps": build_meps,
    }

    if args.dataname == "all":
        order = ["compas", "bank", "lsac", "meps"]
    else:
        order = [args.dataname]

    for name in order:
        print(f"\n=== Building dataset: {name} ===")
        builders[name]("data")

    print("\nAll requested datasets are prepared.")


if __name__ == "__main__":
    main()
