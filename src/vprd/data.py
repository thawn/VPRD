from pathlib import Path
import tables as tb
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

data_path = Path('../../data')
local_data_path = Path(data_path / 'cloud/')
# Calibration factors for time and energy
# see:
# https://codebase.helmholtz.cloud/haicu/vouchers/desy/mirian_virtual-field-diagnostic-tool/-/blob/main/data/Time%20and%20energy%20collibration%20factors%20in%20morning%20shift%2027_02_2024.pdf?ref_type=heads
ENERGY_CALIBRATION_FACTOR = 0.000033 * 879  # MeV/pixel
TIME_CALIBRATION_FACTOR = 0.55  # ± 0.01 fs/pixel
time_energy_aspect_ratio = TIME_CALIBRATION_FACTOR / ENERGY_CALIBRATION_FACTOR


def pixels_to_mev(pixel_data: np.ndarray) -> np.ndarray:
    """
    Calibrates pixel data for energy per electron by multiplying it with a calibration factor.
    The returned energy is in MeV.

    Parameters
    ----------
    pixel_data (np.ndarray): The data to be calibrated.

    Returns
    -------
    np.ndarray: The calibrated energy data in MeV.
    """
    return pixel_data * ENERGY_CALIBRATION_FACTOR


def pixels_to_fs(pixel_data: np.ndarray) -> np.ndarray:
    """
    Calibrates the pixel data for time in fs by multiplying it with a calibration factor.
    The returned time is in fs.

    Parameters:
    pixel_data (np.ndarray): The input time data to be calibrated.

    Returns:
    np.ndarray: The calibrated time data in fs.

    """
    return pixel_data * TIME_CALIBRATION_FACTOR


def mev_to_J(mev: np.ndarray) -> np.ndarray:
    """
    Converts MeV to Joules.

    Parameters
    ----------
    mev (np.ndarray): The energy data to be converted.

    Returns
    -------
    np.ndarray: The energy in J.
    """
    # 1 eV = 1.602176634E-19 J
    return mev * 1.602176634E-13


def mev_to_ev(mev: np.ndarray) -> np.ndarray:
    """
    Converts MeV to eV.

    Parameters
    ----------
    mev (np.ndarray): The energy data to be converted.

    Returns
    -------
    np.ndarray: The energy in eV.
    """
    # 1 MeV = 1e6 eV
    return mev * 1e6


def nanocoulomb_to_coulomb(nanocoulomb: np.ndarray) -> np.ndarray:
    """
    Converts nanocoulomb to coulomb.

    Parameters
    ----------
    nanocoulomb (np.ndarray): The charge in nanocoulomb.

    Returns
    -------
    np.ndarray: The charge in coulomb.
    """
    return nanocoulomb * 1e-9


def nanocoulomb_to_e(nanocoulomb: np.ndarray) -> np.ndarray:
    """
    Converts nanocoulomb to electron charges.

    Parameters
    ----------
    nanocoulomb (np.ndarray): The charge in nanocoulomb.

    Returns
    -------
    np.ndarray: The charge in electron charges.
    """
    # 1 C = 6.241509E18 e
    return nanocoulomb * 6.241509e9


def read_credentials() -> tuple[str, str]:
    """
    Reads the username and password from the 'nextcloud_key.secret' file.

    Returns
    -------
    tuple[str, str]
        A tuple of strings containing the username and password.
    """
    with open(data_path / 'nextcloud_key.secret', 'r') as f:
        username = f.readline().strip()
        password = f.readline().strip()
    return (username, password)


def load_data_from_cloud(target_path: Path):
    import requests
    file_links = {
        "electron_power_data_files_114_115_116_117.hdf5": "https://zenodo.org/records/13738131/files/electron_power_data_files_114_115_116_117.hdf5?download=1&preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcyNTkxMTQzMCwiZXhwIjoxNzM0MjIwNzk5fQ.eyJpZCI6ImViYzE1NzdkLWZmZTItNDQzZi04Zjg2LTllYWJlYmJlZjM0MyIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZGY4OGFkMmJjMzEwNDhiMWEwMzUzNmRkZTc0YTdiNiJ9.22UczbDfqioKIRdJcIo-_VVj4Mdk1kPo6-lPEK_yzPe6-KGInd9HS3afRAiDEoOs0gu9-lo5dgRJcodCG_ST1A",
        "PBD2_VRFD_pbd2_stream_6_run52040_file114_20240227T153103.hdf5": "https://zenodo.org/records/13738131/files/PBD2_VRFD_pbd2_stream_6_run52040_file114_20240227T153103.hdf5?download=1&preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcyNTkxMTQzMCwiZXhwIjoxNzM0MjIwNzk5fQ.eyJpZCI6ImViYzE1NzdkLWZmZTItNDQzZi04Zjg2LTllYWJlYmJlZjM0MyIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZGY4OGFkMmJjMzEwNDhiMWEwMzUzNmRkZTc0YTdiNiJ9.22UczbDfqioKIRdJcIo-_VVj4Mdk1kPo6-lPEK_yzPe6-KGInd9HS3afRAiDEoOs0gu9-lo5dgRJcodCG_ST1A",
        "PBD2_VRFD_pbd2_stream_6_run52040_file115_20240227T153214.hdf5": "https://zenodo.org/records/13738131/files/PBD2_VRFD_pbd2_stream_6_run52040_file115_20240227T153214.hdf5?download=1&preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcyNTkxMTQzMCwiZXhwIjoxNzM0MjIwNzk5fQ.eyJpZCI6ImViYzE1NzdkLWZmZTItNDQzZi04Zjg2LTllYWJlYmJlZjM0MyIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZGY4OGFkMmJjMzEwNDhiMWEwMzUzNmRkZTc0YTdiNiJ9.22UczbDfqioKIRdJcIo-_VVj4Mdk1kPo6-lPEK_yzPe6-KGInd9HS3afRAiDEoOs0gu9-lo5dgRJcodCG_ST1A",
        "PBD2_VRFD_pbd2_stream_6_run52040_file116_20240227T153325.hdf5": "https://zenodo.org/records/13738131/files/PBD2_VRFD_pbd2_stream_6_run52040_file116_20240227T153325.hdf5?download=1&preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcyNTkxMTQzMCwiZXhwIjoxNzM0MjIwNzk5fQ.eyJpZCI6ImViYzE1NzdkLWZmZTItNDQzZi04Zjg2LTllYWJlYmJlZjM0MyIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZGY4OGFkMmJjMzEwNDhiMWEwMzUzNmRkZTc0YTdiNiJ9.22UczbDfqioKIRdJcIo-_VVj4Mdk1kPo6-lPEK_yzPe6-KGInd9HS3afRAiDEoOs0gu9-lo5dgRJcodCG_ST1A",
        "PBD2_VRFD_pbd2_stream_6_run52040_file117_20240227T153436.hdf5": "https://zenodo.org/records/13738131/files/PBD2_VRFD_pbd2_stream_6_run52040_file117_20240227T153436.hdf5?download=1&preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcyNTkxMTQzMCwiZXhwIjoxNzM0MjIwNzk5fQ.eyJpZCI6ImViYzE1NzdkLWZmZTItNDQzZi04Zjg2LTllYWJlYmJlZjM0MyIsImRhdGEiOnt9LCJyYW5kb20iOiI3ZGY4OGFkMmJjMzEwNDhiMWEwMzUzNmRkZTc0YTdiNiJ9.22UczbDfqioKIRdJcIo-_VVj4Mdk1kPo6-lPEK_yzPe6-KGInd9HS3afRAiDEoOs0gu9-lo5dgRJcodCG_ST1A",
    }

    # download all files from the cloud
    r = requests.get(file_links[target_path.name], stream=True)
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True)
    with open(target_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def ensure_local(file_path: Path) -> Path:
    """
    Ensures that the file at the given file path exists locally.
    If the file does not exist, it is loaded from the cloud.

    Parameters
    ----------
    file_path: pathlib.Path
        The path to the file.

    Returns
    -------
    pathlib.Path
        The path to the local file.

    Raises
    ------
    AssertionError
        If the file could not be obtained from the cloud.
    """
    if not file_path.exists():
        load_data_from_cloud(file_path)
    assert file_path.exists(), f"Could not get {file_path} from cloud."
    return file_path


def process_hdf5_data(data: tb.File) -> pd.DataFrame:
    """
    Processes the data from the HDF5 file and returns it as a pandas DataFrame.

    Makes sure that all data is merged on the 'TrainId' column. So that only complete data sets where all data groups have matching 'TrainId's are returned.

    Parameters
    ----------
    data: tb.File
        The tables data file handle containing the measurement data.

    Returns
    -------
    pandas.DataFrame
        The processed data as a DataFrame.
    """
    image_train_id = data.get_node("/FLASH.DIAG/CAMERA/OTR9FL2XTDS", 'TrainId').read()
    images = data.get_node("/FLASH.DIAG/CAMERA/OTR9FL2XTDS", 'Value').read()
    # create a DataFrame with the train IDs and the images
    df = pd.DataFrame({'TrainId': image_train_id, 'Images': [img for img in images]})
    df = add_measurement_data_to_df(df, data)
    return df


def add_measurement_data_to_df(df: pd.DataFrame, data: tb.File) -> pd.DataFrame:
    """
    Adds measurement data from a given `data` hdf5 tables file handle to a DataFrame `df`.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame to which the measurement data will be added.
    data: tb.File
        The tables data file handle containing the measurement data.

    Returns
    -------
    pandas.DataFrame
        The updated DataFrame with the measurement data added.
    """
    # add all other data to the DataFrame, checking that the train IDs are the same
    for group in data.walk_groups():
        if 'TrainId' in data.get_node(where=group) and group._v_pathname != '/FLASH.DIAG/CAMERA/OTR9FL2XTDS':
            data_frame = {}
            for array in data.walk_nodes(where=group, classname="Array"):
                if array._v_name == 'TrainId':
                    data_frame['TrainId'] = array.read()
                else:
                    column_name = array._v_pathname.replace('.', '_')
                    data_frame[column_name] = array.read()
            try:
                temp_df = pd.DataFrame(data_frame)
            except ValueError:
                print(f"Could not create DataFrame for {group}")
                continue

            # merge temp_df with df on TrainID by using `how=inner`, we get rid of incomplete data sets
            df = pd.merge(df, temp_df, on='TrainId', how='inner')
            # check that we did not lose any columns becuase of completely wrong TrainIDs
            for column in temp_df.columns:
                if column not in df.columns:
                    missing_train_ids = temp_df[~temp_df.TrainId.isin(df.TrainId)].TrainId.unique()
                    print(f"Column {column} not added to DataFrame. These train IDs are missing: {missing_train_ids}")

    return df


def calculate_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the time deltas between the arrival times at the DBC and UBC detectors.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame to calculate the time deltas for.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the time deltas added.
    """
    df['/FLASH_SDIAG/BAM_DAQ/FL0_DBC_ARRIVAL_TIME_1_DELTA/Value'] = df['/FLASH_SDIAG/BAM_DAQ/FL0_DBC1_ARRIVAL_TIME_RELATIVE/Value'] - \
        df['/FLASH_SDIAG/BAM_DAQ/FL0_UBC1_ARRIVAL_TIME_RELATIVE/Value']
    df['/FLASH_SDIAG/BAM_DAQ/FL0_DBC_ARRIVAL_TIME_2_DELTA/Value'] = df['/FLASH_SDIAG/BAM_DAQ/FL0_DBC2_ARRIVAL_TIME_RELATIVE/Value'] - \
        df['/FLASH_SDIAG/BAM_DAQ/FL0_UBC2_ARRIVAL_TIME_RELATIVE/Value']
    return df


def hdf5_to_df(file_path: Path) -> pd.DataFrame:
    """
    Reads the specified table from the given HDF5 file and returns it as a pandas DataFrame.

    Parameters
    ----------
    file_path: Path
        The path to the HDF5 file.
    table_name: str
        The name of the table to be read.

    Returns
    -------
    pandas.DataFrame
        The table as a DataFrame.
    """
    ensure_local(file_path)
    with tb.open_file(file_path, 'r') as data:
        df = process_hdf5_data(data)
    df = calculate_time_deltas(df)
    return df


class GlobalStandardScaler:
    """
    A class to scale data globally.

    Parameters
    ----------
    mean: float
        The mean of the data.
    std: float
        The standard deviation of the data.

    Attributes
    ----------
    mean: float
        The mean of the data.
    std: float
        The standard deviation of the data.

    Methods
    -------
    fit(data): Fits the data to the scaler.
    transform(data): Transforms the data using the scaler.
    inverse_transform(data): Inverse transforms the data using the scaler.
    """

    def __init__(self, mean: float = 0, std: float = 1):
        self.mean = mean
        self.std = std

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class FLASHEnergyDataset(Dataset):
    """
    A PyTorch dataset class for FLASHEnergy data.

    Parameters
    ----------
    df: pandas.DataFrame
        The input DataFrame containing the data.

    Attributes
    ----------
    target: np.ndarray
        The target data used as labels during training.
    target_scaler: GlobalStandardScaler
        The target scaler used to scale the target data.
    train_id: np.ndarray
        The train IDs.
    data: np.ndarray
        The input data.
    scaler: StandardScaler
        The input data scaler.
    data_type: np.dtype
        The data type of the input data.
    cut_off: int
        The cut-off point for the target data.

    Methods
    -------
    __len__(): Returns the length of the data.
    __getitem__(index): Returns the data at the given index.
    inverse_transform(data): Inverse transforms the target data using the target_scaler. Useful to scale predictions back to the original scale.
    """

    def __init__(self, df: pd.DataFrame):
        target_column = 'Electron_power'
        self.target = np.array(list(df['Electron_power']))
        # replace nans with 0 in the target
        self.target = np.where(np.isnan(self.target), 0, self.target)
        self.target_scaler = GlobalStandardScaler()
        self.target = self.target_scaler.fit_transform(self.target)
        self.train_id = df['TrainId']
        if '/FLASH_DIAG/BPM/9FL2XTDS/CHARGE_TD' in df.columns and '/FLASH_DIAG/TOROID/7FL2XTDS/CHARGE_TD' in df.columns:
            data = df.drop(
                columns=[
                    target_column,
                    'TrainId',
                    'index',
                    '/FLASH_DIAG/BPM/9FL2XTDS/CHARGE_TD'])
        else:
            data = df.drop(columns=[target_column, 'TrainId', 'index'])
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)
        self.data_type = np.float32
        self.cut_off = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].astype(self.data_type), self.target[index, self.cut_off:].astype(self.data_type)

    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)


def get_train_val_test_dataloader(df: pd.DataFrame, batch_size: int = 0, validation_size: float = 0.1, test_size: float = 0.1,
                                  manual_seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns training and test DataLoaders for the given DataFrame.
    Uses FLASHEnergyDataset as the Dataset class.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to create the DataLoader for.
    batch_size : int
        The batch size to use.
    test_size : float, optional
        The fraction of the data to use for training (default: 0.2).
    manual_seed : int, optional
        The manual seed for random number generation (default: 42).
    device : str, optional
        The device to use for computation (default: 'cpu').

    Returns
    -------
    tuple[DataLoader, DataLoader]
        A tuple containing two DataLoaders: one for training data and one for test data.
    """
    dataset = FLASHEnergyDataset(df)
    generator = torch.Generator().manual_seed(manual_seed)
    lengths = [1 - test_size - validation_size, validation_size, test_size]
    train_data, validation_data, test_data = torch.utils.data.random_split(dataset, lengths, generator=generator)
    if batch_size == 0:
        train_batch_size = len(train_data)
        validation_batch_size = len(validation_data)
        test_batch_size = len(test_data)
    else:
        train_batch_size = batch_size
        validation_batch_size = batch_size
        test_batch_size = batch_size
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_data, batch_size=validation_batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, drop_last=True)
    return train_dataloader, validation_dataloader, test_dataloader


def get_input_shape(dataloader: DataLoader) -> torch.Size:
    """
    Returns the input shape of the data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to get the input shape from.

    Returns
    -------
    int
        The input shape of the data.
    """
    return next(iter(dataloader))[0].shape[1:]


def get_output_shape(dataloader: DataLoader) -> torch.Size:
    """
    Returns the output shape of the data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to get the output shape from.

    Returns
    -------
    int
        The output shape of the data.
    """
    return next(iter(dataloader))[1].shape[1:]
