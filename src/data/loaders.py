import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional

def load_csv(
    f_path: str,
    return_tensors: bool = False,
    chunk_size: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Loads data from a CSV file. If chunk_size is specified and the file is large,
    splits the data into chunks and caches them for later use.

    If the cache_dir is not provided, it creates one based on the CSV file name.
    Default cache directory is `<csv_file_name>_chunks`, placed in the same directory as the CSV file.

    Args:
        f_path (str): Path to the CSV file.
        return_tensors (bool): If True, returns a torch.Tensor, else np.ndarray.
        chunk_size (Optional[int]): Number of rows per chunk. If None, loads all at once.
        cache_dir (Optional[str]): Directory to store chunked files.

    Returns:
        np.ndarray or torch.Tensor: Loaded data.
        If data is chunked, returns a generator yielding chunks from cache_dir.
    """
    if chunk_size is None:
        data = pd.read_csv(f_path).values
        if return_tensors:
            return torch.from_numpy(data.astype(np.float32))
        return data

    if cache_dir is None:
        cache_dir = os.path.splitext(f_path)[0] + "_chunks"
    os.makedirs(cache_dir, exist_ok=True)

    chunk_files = []
    for i, chunk in enumerate(pd.read_csv(f_path, chunksize=chunk_size)):
        chunk_file = os.path.join(cache_dir, f"chunk_{i}.npy")
        if not os.path.exists(chunk_file):
            np.save(chunk_file, chunk.values)
        chunk_files.append(chunk_file)

    def chunk_loader():
        for chunk_file in chunk_files:
            arr = np.load(chunk_file)
            if return_tensors:
                yield torch.from_numpy(arr.astype(np.float32))
            else:
                yield arr

    return chunk_loader

def split_data(
    data: Union[np.ndarray, torch.Tensor],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> Tuple:
    """
    Splits data into train, val, and test sets.
    Default split is 70% train, 15% validation, 15% test.
    Ratios must sum to 1.

    Args:
        data (np.ndarray or torch.Tensor): The data to split.
        ratios (tuple): Ratios for train, val, test.
        seed (int): Random seed.

    Returns:
        Tuple: (train, val, test) splits.
    """
    assert sum(ratios) == 1.0, f"Ratios must sum to 1, got {sum(ratios)}"
    
    n = len(data)
    np.random.seed(seed)
    indices = np.random.permutation(n)
    train_end = int(ratios[0] * n)
    val_end = train_end + int(ratios[1] * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if isinstance(data, torch.Tensor):
        return data[train_idx], data[val_idx], data[test_idx]
    
    return data[train_idx], data[val_idx], data[test_idx]

def load_and_split(
    f_path: str,
    return_tensors: bool = False,
    chunk_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
):
    """
    Loads data from CSV and splits into train, val, test.

    Args:
        f_path (str): Path to CSV.
        return_tensors (bool): Return torch.Tensor if True.
        chunk_size (Optional[int]): Chunk size for large files.
        cache_dir (Optional[str]): Directory for chunked files.
        ratios (tuple): Train/val/test split ratios.
        seed (int): Random seed.

    Returns:
        Tuple: (train, val, test) splits or generator if chunked.
    """
    loader = load_csv(f_path, return_tensors=return_tensors, chunk_size=chunk_size, cache_dir=cache_dir)
    
    # if the loader is a generator (chunked-data)
    if callable(loader):
        def split_chunks():
            for chunk in loader():
                yield split_data(chunk, ratios, seed)
        return split_chunks
    
    return split_data(loader, ratios, seed)  # default data load

class EndOfDataLoader(Exception):
        """Custom exception to signal DataLoader reset without termination."""
        # logic to handle reset can be implemented in the training loop
        pass

class DataLoader:
    """A simple DataLoader for numpy arrays."""
    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        """
        Initializes the DataLoader.

        Args:
            data (np.ndarray): The dataset.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data at the start of each epoch.
        Returns:
            None
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = len(data)
        self.indices = np.arange(self.n)
        self.current_idx = 0
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.n)
        batch_indices = self.indices[start:end]
        
        return self.data[batch_indices]

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        return self
    
    def __next__(self):
        if self.current_idx >= self.n:
            raise EndOfDataLoader
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = self.data[batch_indices]
        self.current_idx += self.batch_size
        
        return batch
    
    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size
    
    def reset(self):
        """
        Resets the DataLoader for a new epoch.
        The current index is set to 0 and data is reshuffled if shuffle is True
        """
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
