import io
import pickle
import numpy as np


def encode_embedding(array):
    """Serialize an embedding using NumPy binary format (pickle-free)."""
    arr = np.asarray(array, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def decode_embedding(blob, return_upgraded_blob=False):
    """
    Deserialize an embedding.

    Supports one-time migration from legacy pickle blobs (magic byte 0x80)
    to NumPy binary bytes. When return_upgraded_blob=True, returns a tuple:
    (array, upgraded_blob, was_upgraded)
    """
    if blob and blob[:1] == b"\x80":
        arr = np.asarray(pickle.loads(blob), dtype=np.float32)
        upgraded_blob = encode_embedding(arr)
        if return_upgraded_blob:
            return arr, upgraded_blob, True
        return arr

    buf = io.BytesIO(blob)
    arr = np.asarray(np.load(buf, allow_pickle=False), dtype=np.float32)
    if return_upgraded_blob:
        return arr, blob, False
    return arr
