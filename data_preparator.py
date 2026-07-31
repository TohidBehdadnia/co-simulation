import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def find_files(folder: Path, extension: str) -> list[Path]:
    return sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix == extension],
        key=lambda p: p.name,
    )

def convert(fileData: np.lib.npyio.NpzFile, delay: float) -> pd.DataFrame:
    tripTime = 1 # The trip event happened at 1 second
    # keys ['t', 'freq', 'measured_delay', 'last_cmd_time', 'buses', 'per_bus_delay']
    t = fileData["t"] - tripTime
    freq = fileData["freq"]
    # Add a feature relative to the number of setpoints received up to now (count the number of elements in measured_delay that are less than or equal to the current time)
    measured_delay = fileData["per_bus_delay"]
    setpoint_count = np.array([np.sum(measured_delay <= time) if np.sum(measured_delay <= time) > 0 else 0 for time in t])
    # Add a feature for the delay of the average delay of the commands up to the current time
    average_delay = np.array([np.mean(measured_delay[measured_delay <= time]) if np.sum(measured_delay <= time) > 0 else 0 for time in t])

    # Compute lag features using timestamp lookups.
    freq_lag_50ms = np.zeros(len(t))
    freq_lag_100ms = np.zeros(len(t))

    idx_50 = np.searchsorted(t, t - 0.05, side="right") - 1
    valid_50 = idx_50 >= 0
    freq_lag_50ms[valid_50] = freq[idx_50[valid_50]]

    idx_100 = np.searchsorted(t, t - 0.1, side="right") - 1
    valid_100 = idx_100 >= 0
    freq_lag_100ms[valid_100] = freq[idx_100[valid_100]]

    return pd.DataFrame({
        "t-from-trip": t,
        "freq": freq,
        "freq-lag-50ms": freq_lag_50ms,
        "freq-lag-100ms": freq_lag_100ms,
        "setpoint-count": setpoint_count,
        "average-delay": average_delay
    })


bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
dataFolder = Path("./data").absolute()
resultsFolder = Path("./results").absolute()
classToFolders = {
    "normal": ["case_iec104_direct_percmd"],
    "delay": ["case_iec104_mitm_delay_dmin0p4_dmax0p6_percmd", "case_iec104_mitm_delay_dmin0p15_dmax0p35_percmd", "case_iec104_mitm_delay_dmin0p65_dmax0p85_percmd"],
    "drop": ["case_iec104_mitm_drop_pdrop0p1_percmd", "case_iec104_mitm_drop_pdrop0p5_percmd", "case_iec104_mitm_drop_pdrop0p9_percmd"],
    "mod": ["case_iec104_mitm_mod_pmod0p1_f1p5_percmd", "case_iec104_mitm_mod_pmod0p5_f1p5_percmd", "case_iec104_mitm_mod_pmod0p9_f1p5_percmd"]
}

def main():
    print("Starting data preparation...")
    classToFoldersPaths = {c: [resultsFolder / f for f in folders] for c, folders in classToFolders.items()}
    # Find all files with ".npz" extension in the folders and associate it with the class
    classToFilesPaths = {c: [f for folder in folders for f in find_files(folder, ".npz")] for c, folders in classToFoldersPaths.items()}

    # Load the data from the .npz files and create a DataFrame
    classToData = {}

    print("Loading data from .npz files...")
    for c, files in tqdm(classToFilesPaths.items(), desc="Loading data for all classes", total=len(classToFilesPaths), bar_format=bar_format):
        classDataDf = [] 
        for filePath in tqdm(files, desc=f"Loading files for class '{c}'", leave=False, total=len(files), bar_format=bar_format):
            df = np.load(filePath)
            classDataDf.append(df)
        classToData[c] = classDataDf

    finalDfColumns = ["t-from-trip", "freq", "freq-lag-50ms", "freq-lag-100ms", "setpoint-count", "average-delay", "run-id", "class"]
    finalDfParts = []
    runId = 0
    for c, dataList in tqdm(classToData.items(), desc="Processing data for all classes", total=len(classToData), bar_format=bar_format):
        for data in tqdm(dataList, desc=f"Processing runs for class '{c}'", leave=False, total=len(dataList), bar_format=bar_format):
            runId += 1
            dataDf = convert(data, delay=0.0001)
            dataDf["class"] = c
            dataDf["run-id"] = runId
            # Remove the rows where t-from-trip is negative (before the trip event)
            dataDf = dataDf[dataDf["t-from-trip"] >= 0]

            selectedColumnsDf = dataDf[finalDfColumns]
            if not selectedColumnsDf.empty:
                finalDfParts.append(selectedColumnsDf)

        finalDf = pd.concat(finalDfParts, ignore_index=True) if finalDfParts else pd.DataFrame(columns=finalDfColumns)

    print("Data preparation completed. Final DataFrame shape:", finalDf.shape)

    # Report class balance in terms of distinct runs and samples.
    run_counts = finalDf.groupby("class")["run-id"].nunique().sort_index()
    sample_counts = finalDf["class"].value_counts().sort_index()
    print("Run counts by class:")
    print(run_counts.to_string())
    print("Sample counts by class:")
    print(sample_counts.to_string())

    print(finalDf.head(n=10))

    finalDf.to_csv(dataFolder / "final_data.csv", index=False)


if __name__ == "__main__":
    main()