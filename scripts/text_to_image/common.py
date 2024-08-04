import importlib
import io
import json
import os
import struct
import zipfile
import zlib

import bisect
import numpy as np
import torch
import transforms3d

class PartialReadableRawIO(io.RawIOBase):
    def __init__(
        self, base_io_object: io.RawIOBase, start: int, end: int,
        close_with_this_object: bool = False
    ):
        super().__init__()
        self.base_io_object = base_io_object
        self.p = self.start = start
        self.end = end
        self.close_with_this_object = close_with_this_object
        self.base_io_object.seek(start)

    def close(self):
        if self.close_with_this_object:
            self.base_io_object.close()

    @property
    def closed(self):
        return self.base_io_object.closed if self.close_with_this_object \
            else False

    def readable(self):
        return self.base_io_object.readable()

    def read(self, size=-1):
        read_count = min(size, self.end - self.p) \
            if size >= 0 else self.end - self.p
        data = self.base_io_object.read(read_count)
        self.p += read_count
        return data

    def readall(self):
        return self.read(-1)

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            p = max(0, min(self.end - self.start, offset))
        elif whence == os.SEEK_CUR:
            p = max(
                0, min(self.end - self.start, self.p - self.start + offset))
        elif whence == os.SEEK_END:
            p = max(
                0, min(self.end - self.start, self.end - self.start + offset))

        self.p = self.base_io_object.seek(self.start + p, os.SEEK_SET)
        return self.p

    def seekable(self):
        return self.base_io_object.seekable()

    def tell(self):
        return self.p - self.start

    def writable(self):
        return False


class LazyFile():
    def __init__(self, path: str, mode: str = "rb"):
        self.path = path
        self.mode = mode

    def open(self, **kwargs):
        return open(self.path, self.mode, **kwargs)


class StatelessZipFile():
    def __init__(self, lazy_file, cache_path=None):
        self.lazy_file = lazy_file

        if cache_path is not None:
            with open(cache_path, "r", encoding="utf-8") as f:
                self.items = json.load(f)

        else:
            with self.lazy_file.open() as f:
                with zipfile.ZipFile(f) as zf:
                    self.items = {
                        i.filename: i.header_offset
                        for i in zf.infolist()
                    }

    def namelist(self):
        return list(self.items.keys())

    def read(self, name: str):
        header_offset = self.items[name]
        with self.lazy_file.open() as f:
            f.seek(header_offset)
            fh = struct.unpack(zipfile.structFileHeader, f.read(30))
            offset = header_offset + 30 + fh[zipfile._FH_FILENAME_LENGTH] + \
                fh[zipfile._FH_EXTRA_FIELD_LENGTH]
            size = fh[zipfile._FH_COMPRESSED_SIZE]
            method = fh[zipfile._FH_COMPRESSION_METHOD]

            f.seek(offset)
            data = f.read(size)

        if method == zipfile.ZIP_STORED:
            return data
        elif method == zipfile.ZIP_DEFLATED:
            return zlib.decompress(data, -15)
        else:
            raise NotImplementedError(
                "That compression method is not supported")

    def get_io_object(self, name: str):
        header_offset = self.items[name]
        f = self.lazy_file.open()
        f.seek(header_offset)
        fh = struct.unpack(zipfile.structFileHeader, f.read(30))
        method = fh[zipfile._FH_COMPRESSION_METHOD]
        assert method == zipfile.ZIP_STORED

        offset = header_offset + 30 + fh[zipfile._FH_FILENAME_LENGTH] + \
            fh[zipfile._FH_EXTRA_FIELD_LENGTH]
        size = fh[zipfile._FH_COMPRESSED_SIZE]
        return PartialReadableRawIO(f, offset, offset + size, True)


class ChainedReaders():
    def __init__(self, reader_list: list):
        self.reader_list = reader_list
        self.dict = {
            j: i_id
            for i_id, i in enumerate(reader_list)
            for j in i.namelist()
        }

    def namelist(self):
        return list([
            j for i in self.reader_list
            for j in i.namelist()
        ])

    def read(self, name: str):
        reader_id = self.dict[name]
        return self.reader_list[reader_id].read(name)

    def get_io_object(self, name: str):
        reader_id = self.dict[name]
        return self.reader_list[reader_id].get_io_object(name)


def get_class(class_name: str):
    if "." in class_name:
        i = class_name.rfind(".")
        module_name = class_name[:i]
        class_name = class_name[i+1:]

        module_type = importlib.import_module(module_name, package=None)
        class_type = getattr(module_type, class_name)
    elif class_name in globals():
        class_type = globals()[class_name]
    else:
        raise RuntimeError("Failed to find the class {}.".format(class_name))

    return class_type


def create_instance(class_name: str, **kwargs):
    class_type = get_class(class_name)
    return class_type(**kwargs)


def create_instance_from_config(_config: dict, level: int = 0, **kwargs):
    if isinstance(_config, dict):
        if "_class_name" in _config:
            args = instantiate_config(_config, level)
            if level == 0:
                args.update(kwargs)

            if _config["_class_name"] == "get_class":
                return get_class(**args)
            else:
                return create_instance(_config["_class_name"], **args)

        else:
            return instantiate_config(_config, level)

    elif isinstance(_config, list):
        return [create_instance_from_config(i, level + 1) for i in _config]
    else:
        return _config


def instantiate_config(_config: dict, level: int = 0):
    return {
        k: create_instance_from_config(v, level + 1)
        for k, v in _config.items() if k != "_class_name"
    }


class Copy():
    def __call__(self, a):
        return a


class DatasetAdapter(torch.utils.data.Dataset):
    def apply_transform(transform, a, stack: bool = True):
        if isinstance(a, list):
            result = [
                DatasetAdapter.apply_transform(transform, i, stack) for i in a
            ]
            if stack:
                result = torch.stack(result)

            return result
        else:
            return transform(a)

    def apply_temporal_transform(transform, a):
        return transform(a)

    def __init__(
        self, base_dataset: torch.utils.data.Dataset, transform_list: list,
        pop_list=None
    ):
        self.base_dataset = base_dataset
        self.transform_list = transform_list
        self.pop_list = pop_list

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        item = self.base_dataset[index]
        for i in self.transform_list:
            if getattr(i["transform"], 'is_temporal_transform', False):
                item[i["new_key"]] = DatasetAdapter.apply_temporal_transform(
                    i["transform"], item[i["old_key"]])
            else:
                item[i["new_key"]] = DatasetAdapter.apply_transform(
                    i["transform"], item[i["old_key"]],
                    i["stack"] if "stack" in i else True)

        if self.pop_list is not None:
            for i in self.pop_list:
                if i in item:
                    item.pop(i)

        return item


class CollateFnIgnoring():
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, item_list: list):
        ignored = [
            (key, [item.pop(key) for item in item_list])
            for key in self.keys
        ]
        result = torch.utils.data.default_collate(item_list)
        for key, value in ignored:
            result[key] = value

        return result


def get_collate_fn_ignoring(keys: list):

    def collate_fn(item_list: list):
        ignored = [
            (key, [item.pop(key) for item in item_list]) for key in keys
        ]
        result = torch.utils.data.default_collate(item_list)
        for key, value in ignored:
            result[key] = value

        return result

    return collate_fn


def find_sample_data_of_nearest_time(
    sample_data_list: list, timestamp_list: list, timestamp
):
    i = bisect.bisect_left(timestamp_list, timestamp)
    t0 = timestamp - timestamp_list[i - 1]
    if i >= len(timestamp_list):
        i = len(timestamp_list) - 1
    else:
        t1 = timestamp_list[i] - timestamp
        if i > 0 and t0 <= t1:
            i -= 1

    return i if sample_data_list is None else sample_data_list[i]


def get_transform(rotation: list, translation: list, output_type: str = "np"):
    result = np.eye(4)
    result[:3, :3] = transforms3d.quaternions.quat2mat(rotation)
    result[:3, 3] = np.array(translation)
    if output_type == "np":
        return result
    elif output_type == "pt":
        return torch.tensor(result, dtype=torch.float32)
    else:
        raise Exception("Unknown output type of the get_transform()")


def make_intrinsic_matrix(fx_fy: list, cx_cy: list, output_type: str = "np"):
    result = np.diag(fx_fy + [1])
    result[:2, 2] = np.array(cx_cy)
    if output_type == "np":
        return result
    elif output_type == "pt":
        return torch.tensor(result, dtype=torch.float32)
    else:
        raise Exception("Unknown output type of the make_intrinsic_matrix()")


def project_line(
    a: np.array, b: np.array, near_z: float = 0.05, far_z: float = 512.0
):
    if (a[2] < near_z and b[2] < near_z) or (a[2] > far_z and b[2] > far_z):
        return None

    ca = a
    cb = b
    if a[2] >= near_z and b[2] < near_z:
        r = (near_z - b[2]) / (a[2] - b[2])
        cb = a * r + b * (1 - r)
    elif a[2] < near_z and b[2] >= near_z:
        r = (b[2] - near_z) / (b[2] - a[2])
        ca = a * r + b * (1 - r)

    if a[2] > far_z and b[2] <= far_z:
        r = (far_z - b[2]) / (a[2] - b[2])
        ca = a * r + b * (1 - r)
    elif a[2] <= far_z and b[2] > far_z:
        r = (b[2] - far_z) / (b[2] - a[2])
        cb = a * r + b * (1 - r)

    pa = ca[:2] / ca[2]
    pb = cb[:2] / cb[2]
    return (pa[0], pa[1], pb[0], pb[1])
