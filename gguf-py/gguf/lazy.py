from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod

from io import BufferedReader, BufferedWriter
from pathlib import Path
from typing import Any, Callable, Iterable

import logging
import numpy as np
import os
import shutil

from numpy.typing import DTypeLike

from .utility import LocalTensorRange


logger = logging.getLogger(__name__)


class LazyMeta(ABCMeta):

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs):
        def __getattr__(self, name: str) -> Any:
            meta_attr = getattr(self._meta, name)
            if callable(meta_attr):
                return type(self)._wrap_fn(
                    (lambda s, *args, **kwargs: getattr(s, name)(*args, **kwargs)),
                    use_self=self,
                    data_noop=name in ("view", "reshape", "squeeze", "unsqueeze", "contiguous"),
                )
            elif isinstance(meta_attr, self._tensor_type):
                # e.g. self.T with torch.Tensor should still be wrapped
                return type(self)._wrap_fn(lambda s: getattr(s, name), use_self=self)()
            else:
                # no need to wrap non-tensor properties,
                # and they likely don't depend on the actual contents of the tensor
                return meta_attr

        namespace["__getattr__"] = __getattr__

        # need to make a builder for the wrapped wrapper to copy the name,
        # or else it fails with very cryptic error messages,
        # because somehow the same string would end up in every closures
        def mk_wrap(op_name: str, *, meta_noop: bool = False):
            # need to wrap the wrapper to get self
            def wrapped_special_op(self, *args, **kwargs):
                return type(self)._wrap_fn(
                    getattr(type(self)._tensor_type, op_name),
                    use_self=self,
                    meta_noop=meta_noop,
                )(*args, **kwargs)
            return wrapped_special_op

        # special methods bypass __getattr__, so they need to be added manually
        # ref: https://docs.python.org/3/reference/datamodel.html#special-lookup
        # NOTE: doing this from a metaclass is very convenient
        # TODO: make this even more comprehensive
        for binary_op in (
            "lt", "le", "eq", "ne", "ge", "gt", "not"
            "abs", "add", "and", "floordiv", "invert", "lshift", "mod", "mul", "matmul",
            "neg", "or", "pos", "pow", "rshift", "sub", "truediv", "xor",
            "iadd", "iand", "ifloordiv", "ilshift", "imod", "imul", "ior", "irshift", "isub", "ixor",
            "radd", "rand", "rfloordiv", "rmul", "ror", "rpow", "rsub", "rtruediv", "rxor",
        ):
            attr_name = f"__{binary_op}__"
            # the result of these operators usually has the same shape and dtype as the input,
            # so evaluation on the meta tensor can be skipped.
            namespace[attr_name] = mk_wrap(attr_name, meta_noop=True)

        for special_op in (
            "getitem", "setitem", "len",
        ):
            attr_name = f"__{special_op}__"
            namespace[attr_name] = mk_wrap(attr_name, meta_noop=False)

        return super().__new__(cls, name, bases, namespace, **kwargs)


# Tree of lazy tensors
class LazyBase(ABC, metaclass=LazyMeta):
    _tensor_type: type
    _meta: Any
    _data: Any | None
    _args: tuple
    _kwargs: dict[str, Any]
    _func: Callable[[Any], Any] | None
    _ranges: tuple[LocalTensorRange, ...]

    def __init__(self, *, meta: Any, data: Any | None = None, args: tuple = (), kwargs: dict[str, Any] | None = None, func: Callable[[Any], Any] | None = None, ranges: tuple[LocalTensorRange, ...] = ()):
        super().__init__()
        self._meta = meta
        self._data = data
        self._args = args
        self._kwargs = kwargs if kwargs is not None else {}
        self._func = func
        self._ranges = ranges
        assert self._func is not None or self._data is not None

    def __init_subclass__(cls) -> None:
        if "_tensor_type" not in cls.__dict__:
            raise TypeError(f"property '_tensor_type' must be defined for {cls!r}")
        return super().__init_subclass__()

    @staticmethod
    def _recurse_apply(o: Any, fn: Callable[[Any], Any]) -> Any:
        # TODO: dict and set
        if isinstance(o, (list, tuple)):
            L = []
            for item in o:
                L.append(LazyBase._recurse_apply(item, fn))
            if isinstance(o, tuple):
                L = tuple(L)
            return L
        elif isinstance(o, LazyBase):
            return fn(o)
        else:
            return o

    @classmethod
    def _wrap_fn(cls, fn: Callable, *, use_self: LazyBase | None = None, meta_noop: bool | DTypeLike | tuple[DTypeLike, Callable[[tuple[int, ...]], tuple[int, ...]]] = False, data_noop: bool = False) -> Callable[[Any], Any]:
        def wrapped_fn(*args, **kwargs):
            if kwargs is None:
                kwargs = {}
            args = ((use_self,) if use_self is not None else ()) + args

            meta_args = LazyBase._recurse_apply(args, lambda t: t._meta)
            # TODO: maybe handle tensors in kwargs too

            ranges = use_self._ranges if use_self is not None and data_noop else ()

            if isinstance(meta_noop, bool) and not meta_noop:
                try:
                    res = fn(*meta_args, **kwargs)
                except NotImplementedError:
                    # running some operations on PyTorch's Meta tensors can cause this exception
                    res = None
            else:
                # some operators don't need to actually run on the meta tensors
                assert len(args) > 0
                res = args[0]
                assert isinstance(res, cls)
                res = res._meta
                # allow operations to override the dtype and shape
                if meta_noop is not True:
                    if isinstance(meta_noop, tuple):
                        dtype, shape = meta_noop
                        assert callable(shape)
                        res = cls.meta_with_dtype_and_shape(dtype, shape(res.shape))
                    else:
                        res = cls.meta_with_dtype_and_shape(meta_noop, res.shape)

            if isinstance(res, cls._tensor_type):
                return cls(meta=cls.eager_to_meta(res), args=args, kwargs=kwargs, func=fn, ranges=ranges)
            elif isinstance(res, tuple) and all(isinstance(t, cls._tensor_type) for t in res):
                # share the evaluation between lazy tuple elements
                shared_args: list = [args, None]

                def eager_tuple_element(a: list[Any], i: int = 0, /, **kw) -> LazyBase:
                    assert len(a) == 2
                    if a[1] is None:
                        a[1] = fn(*a[0], **kw)
                    return a[1][i]
                return tuple(cls(meta=cls.eager_to_meta(res[i]), args=(shared_args, i), kwargs=kwargs, func=eager_tuple_element) for i in range(len(res)))
            else:
                del res  # not needed
                # non-tensor return likely relies on the contents of the args
                # (e.g. the result of torch.equal)
                eager_args = cls.to_eager(args)
                return fn(*eager_args, **kwargs)
        return wrapped_fn

    @classmethod
    def to_eager(cls, t: Any) -> Any:
        def simple_to_eager(_t: LazyBase) -> Any:
            if _t._data is not None:
                return _t._data

            # NOTE: there's a recursion limit in Python (usually 1000)

            assert _t._func is not None
            _t._args = cls._recurse_apply(_t._args, simple_to_eager)
            _t._data = _t._func(*_t._args, **_t._kwargs)
            # sanity check
            assert _t._data is not None
            assert _t._data.dtype == _t._meta.dtype
            assert _t._data.shape == _t._meta.shape

            return _t._data

        # recurse into lists and/or tuples, keeping their structure
        return cls._recurse_apply(t, simple_to_eager)

    @classmethod
    def eager_to_meta(cls, t: Any) -> Any:
        return cls.meta_with_dtype_and_shape(t.dtype, t.shape)

    # must be overridden, meta tensor init is backend-specific
    @classmethod
    @abstractmethod
    def meta_with_dtype_and_shape(cls, dtype: Any, shape: Any) -> Any: pass

    @classmethod
    def from_eager(cls, t: Any) -> Any:
        if type(t) is cls:
            # already lazy
            return t
        elif isinstance(t, cls._tensor_type):
            return cls(meta=cls.eager_to_meta(t), data=t)
        else:
            return TypeError(f"{type(t)!r} is not compatible with {cls._tensor_type!r}")


class LazyNumpyTensor(LazyBase):
    _tensor_type = np.ndarray

    shape: tuple[int, ...]  # Makes the type checker happy in quants.py
    nbytes: int

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: DTypeLike, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
        # The initial idea was to use np.nan as the fill value,
        # but non-float types like np.int16 can't use that.
        # So zero it is.
        cheat = np.zeros(1, dtype)
        return np.lib.stride_tricks.as_strided(cheat, shape, (0 for _ in shape))

    def astype(self, dtype, *args, **kwargs):
        meta = type(self).meta_with_dtype_and_shape(dtype, self._meta.shape)
        full_args = (self, dtype,) + args
        ranges = self._ranges if self._meta.dtype == dtype else ()
        return type(self)(meta=meta, args=full_args, kwargs=kwargs, func=(lambda a, *args, **kwargs: a.astype(*args, **kwargs)), ranges=ranges)

    def tofile(self, fid, *args, **kwargs):
        if isinstance(fid, BufferedWriter) and len(self._ranges) > 0:
            return copy_tensor_ranges(self, fid)
        else:
            eager = LazyNumpyTensor.to_eager(self)
            return eager.tofile(fid, *args, **kwargs)

    # TODO: __array_function__


# For aligning blocks when reflinking
def best_extra_offset(t: np.ndarray | LazyNumpyTensor | None, current_offset: int) -> int:
    if not isinstance(t, LazyNumpyTensor):
        # no file ranges, no need for an offset
        return 0

    ranges = t._ranges

    histogram: dict[int, int] = {}

    max_block_size = 0
    for r in ranges:
        # Ensure minimal alignment is 8 bytes (common with safetensors)
        # and that the block size is valid
        if r.offset % 8 == 0 and r.block_size > 0:
            align_offset = r.offset % r.block_size
            if align_offset not in histogram:
                histogram[align_offset] = 0
            histogram[align_offset] += r.size
            if r.block_size > max_block_size:
                max_block_size = r.block_size

    best_offset = 0
    best_size = 0
    for offset, size in histogram.items():
        if size > best_size:
            best_size = size
            best_offset = offset

    if max_block_size > 0:
        # the offset needs to be aligned properly
        # or else there's probably a block size mismatch
        assert current_offset % max_block_size == 0, current_offset % max_block_size

    return best_offset


def count_reflinkable_size(tensors: Iterable[tuple[str, np.ndarray | LazyNumpyTensor | None]]) -> int:
    if not hasattr(os, "copy_file_range"):
        return 0
    size = 0
    for name, t in tensors:
        if isinstance(t, LazyNumpyTensor) and len(t._ranges) > 0:
            align_offset = best_extra_offset(t, 0)
            misaligned = 0
            for range in t._ranges:
                if range.block_size > 0:
                    if range.offset % range.block_size == align_offset:
                        size += range.size
                    else:
                        misaligned += 1
            if misaligned > 0:
                logger.debug(f"{name} misaligned for reflinking, fallback to copy for {misaligned} of {len(t._ranges)} parts")
    return size


# Copy tensor ranges using os.copy_file_range with aligned offsets and sizes
# to make it more likely that copy-on-write is used where possible.
# Block alignment is necessary for BTRFS and XFS (and likely for ZFS too).
#
# Falls back to shutil.copyfileobj when os.copy_file_range is not present.
def copy_tensor_ranges(t: LazyNumpyTensor, fout: BufferedWriter):
    ranges = t._ranges
    assert len(ranges) > 0
    dst_offset = fout.tell()
    extra_offset = best_extra_offset(t, dst_offset)

    if extra_offset > 0:
        # initial padding
        fout.write(b"\x00" * extra_offset)

    dst_offset += extra_offset
    start_offset = dst_offset

    src_files: dict[Path, BufferedReader] = {}
    for r in ranges:
        if r.filename not in src_files:
            src_files[r.filename] = open(r.filename, "rb")

    has_copy_file_range = hasattr(os, "copy_file_range")

    for r in ranges:
        src = src_files[r.filename]
        if has_copy_file_range:
            if r.block_size > 0 and (r.offset % r.block_size) == (start_offset % r.block_size):
                # Attempting to align copies for reflinking

                # Block  0,      1,      2,      3,      4,
                # |___0000|0000000|0001111|1111111|111____|
                #
                # 1. block 0 is partially overwritten with contents from range[0]
                # 2. blocks 1 and 2 are copied from range[0] using os.copy_file_range
                # 3. block 2 is partially overwritten with contents from range[1]
                # 4. blocks 3 and 4 are copied from range[1] using os.copy_file_range
                # (repeated for further ranges)
                if dst_offset % r.block_size == 0:
                    extra_size = 0
                else:
                    extra_size = r.block_size - (dst_offset % r.block_size)
                    extra_size = min(extra_size, r.size)
                    src.seek(r.offset)
                    buf = src.read(extra_size)
                    fout.seek(dst_offset)
                    fout.write(buf)
                    dst_offset += extra_size
                    if extra_size == r.size:
                        continue

                assert dst_offset % r.block_size == 0, dst_offset % r.block_size

                offset_src = r.offset + extra_size
                offset_src_end = r.offset + r.size
                if offset_src_end % r.block_size != 0:
                    offset_src_end += r.block_size - (offset_src_end % r.block_size)
                size = offset_src_end - offset_src
                os.copy_file_range(src.fileno(), fout.fileno(), size, offset_src, dst_offset)
                dst_offset += r.size - extra_size
            else:
                # not trying to use reflinks, but still using os.copy_file_range for speed
                os.copy_file_range(src.fileno(), fout.fileno(), r.size, r.offset, dst_offset)
                dst_offset += r.size
        else:
            # not using reflinks, fallback when os.copy_file_range is not supported
            src.seek(r.offset)
            fout.seek(dst_offset)
            shutil.copyfileobj(src, fout, r.size)
            dst_offset += r.size

    for f in src_files.values():
        f.close()

    fout.seek(dst_offset)
