#!/usr/bin/env python3

import logging
import argparse
import heapq
import sys
import os
from glob import glob
import sqlite3
import json
import csv
from typing import Optional, Union
from collections.abc import Iterator, Sequence

try:
    import git
    from tabulate import tabulate
except ImportError as e:
    print("the following Python libraries are required: GitPython, tabulate.") # noqa: NP100
    raise e


logger = logging.getLogger("compare-llama-bench")

# All llama-bench SQL fields
DB_FIELDS = [
    "build_commit", "build_number", "cpu_info",       "gpu_info",   "backends",     "model_filename",
    "model_type",   "model_size",   "model_n_params", "n_batch",    "n_ubatch",     "n_threads",
    "cpu_mask",     "cpu_strict",   "poll",           "type_k",     "type_v",       "n_gpu_layers",
    "split_mode",   "main_gpu",     "no_kv_offload",  "flash_attn", "tensor_split", "tensor_buft_overrides",
    "defrag_thold",
    "use_mmap",     "embeddings",   "no_op_offload",  "n_prompt",   "n_gen",        "n_depth",
    "test_time",    "avg_ns",       "stddev_ns",      "avg_ts",     "stddev_ts",
]

DB_TYPES = [
    "TEXT",    "INTEGER", "TEXT",    "TEXT",    "TEXT",    "TEXT",
    "TEXT",    "INTEGER", "INTEGER", "INTEGER", "INTEGER", "INTEGER",
    "TEXT",    "INTEGER", "INTEGER", "TEXT",    "TEXT",    "INTEGER",
    "TEXT",    "INTEGER", "INTEGER", "INTEGER", "TEXT",    "TEXT",
    "REAL",
    "INTEGER", "INTEGER", "INTEGER", "INTEGER", "INTEGER", "INTEGER",
    "TEXT",    "INTEGER", "INTEGER", "REAL",    "REAL",
]
assert len(DB_FIELDS) == len(DB_TYPES)

# Properties by which to differentiate results per commit:
KEY_PROPERTIES = [
    "cpu_info", "gpu_info", "backends", "n_gpu_layers", "tensor_buft_overrides", "model_filename", "model_type",
    "n_batch", "n_ubatch", "embeddings", "cpu_mask", "cpu_strict", "poll", "n_threads", "type_k", "type_v",
    "use_mmap", "no_kv_offload", "split_mode", "main_gpu", "tensor_split", "flash_attn", "n_prompt", "n_gen", "n_depth"
]

# Properties that are boolean and are converted to Yes/No for the table:
BOOL_PROPERTIES = ["embeddings", "cpu_strict", "use_mmap", "no_kv_offload", "flash_attn"]

# Header names for the table:
PRETTY_NAMES = {
    "cpu_info": "CPU", "gpu_info": "GPU", "backends": "Backends", "n_gpu_layers": "GPU layers",
    "tensor_buft_overrides": "Tensor overrides", "model_filename": "File", "model_type": "Model", "model_size": "Model size [GiB]",
    "model_n_params": "Num. of par.", "n_batch": "Batch size", "n_ubatch": "Microbatch size", "embeddings": "Embeddings",
    "cpu_mask": "CPU mask", "cpu_strict": "CPU strict", "poll": "Poll", "n_threads": "Threads", "type_k": "K type", "type_v": "V type",
    "use_mmap": "Use mmap", "no_kv_offload": "NKVO", "split_mode": "Split mode", "main_gpu": "Main GPU", "tensor_split": "Tensor split",
    "flash_attn": "FlashAttention",
}

DEFAULT_SHOW = ["model_type"]  # Always show these properties by default.
DEFAULT_HIDE = ["model_filename"]  # Always hide these properties by default.
GPU_NAME_STRIP = ["NVIDIA GeForce ", "Tesla ", "AMD Radeon "]  # Strip prefixes for smaller tables.
MODEL_SUFFIX_REPLACE = {" - Small": "_S", " - Medium": "_M", " - Large": "_L"}

DESCRIPTION = """Creates tables from llama-bench data written to multiple JSON/CSV files, a single JSONL file or SQLite database. Example usage (Linux):

$ git checkout master
$ make clean && make llama-bench
$ ./llama-bench -o sql | sqlite3 llama-bench.sqlite
$ git checkout some_branch
$ make clean && make llama-bench
$ ./llama-bench -o sql | sqlite3 llama-bench.sqlite
$ ./scripts/compare-llama-bench.py

Performance numbers from multiple runs per commit are averaged WITHOUT being weighted by the --repetitions parameter of llama-bench.
"""

parser = argparse.ArgumentParser(
    description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
help_b = (
    "The baseline commit to compare performance to. "
    "Accepts either a branch name, tag name, or commit hash. "
    "Defaults to latest master commit with data."
)
parser.add_argument("-b", "--baseline", help=help_b)
help_c = (
    "The commit whose performance is to be compared to the baseline. "
    "Accepts either a branch name, tag name, or commit hash. "
    "Defaults to the non-master commit for which llama-bench was run most recently."
)
parser.add_argument("-c", "--compare", help=help_c)
help_i = (
    "JSON/JSONL/SQLite/CSV files for comparing commits. "
    "Specify multiple times to use multiple input files (JSON/CSV only). "
    "Defaults to 'llama-bench.sqlite' in the current working directory. "
    "If no such file is found and there is exactly one .sqlite file in the current directory, "
    "that file is instead used as input."
)
parser.add_argument("-i", "--input", action="append", help=help_i)
help_o = (
    "Output format for the table. "
    "Defaults to 'pipe' (GitHub compatible). "
    "Also supports e.g. 'latex' or 'mediawiki'. "
    "See tabulate documentation for full list."
)
parser.add_argument("-o", "--output", help=help_o, default="pipe")
help_s = (
    "Columns to add to the table. "
    "Accepts a comma-separated list of values. "
    f"Legal values: {', '.join(KEY_PROPERTIES[:-3])}. "
    "Defaults to model name (model_type) and CPU and/or GPU name (cpu_info, gpu_info) "
    "plus any column where not all data points are the same. "
    "If the columns are manually specified, then the results for each unique combination of the "
    "specified values are averaged WITHOUT weighing by the --repetitions parameter of llama-bench."
)
parser.add_argument("--check", action="store_true", help="check if all required Python libraries are installed")
parser.add_argument("-s", "--show", help=help_s)
parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
parser.add_argument("--plot", help="generate a performance comparison plot and save to specified file (e.g., plot.png)")
parser.add_argument("--plot_x", help="parameter to use as x axis for plotting (default: n_depth)", default="n_depth")
parser.add_argument("--plot_log_scale", action="store_true", help="use log scale for x axis in plots (off by default)")

known_args, unknown_args = parser.parse_known_args()

logging.basicConfig(level=logging.DEBUG if known_args.verbose else logging.INFO)


if known_args.check:
    # Check if all required Python libraries are installed. Would have failed earlier if not.
    sys.exit(0)

if unknown_args:
    logger.error(f"Received unknown args: {unknown_args}.\n")
    parser.print_help()
    sys.exit(1)

input_file = known_args.input
if not input_file and os.path.exists("./llama-bench.sqlite"):
    input_file = ["llama-bench.sqlite"]
if not input_file:
    sqlite_files = glob("*.sqlite")
    if len(sqlite_files) == 1:
        input_file = sqlite_files

if not input_file:
    logger.error("Cannot find a suitable input file, please provide one.\n")
    parser.print_help()
    sys.exit(1)


class LlamaBenchData:
    repo: Optional[git.Repo]
    build_len_min: int
    build_len_max: int
    build_len: int = 8
    builds: list[str] = []
    check_keys = set(KEY_PROPERTIES + ["build_commit", "test_time", "avg_ts"])

    def __init__(self):
        try:
            self.repo = git.Repo(".", search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            self.repo = None

    def _builds_init(self):
        self.build_len = self.build_len_min

    def _check_keys(self, keys: set) -> Optional[set]:
        """Private helper method that checks against required data keys and returns missing ones."""
        if not keys >= self.check_keys:
            return self.check_keys - keys
        return None

    def find_parent_in_data(self, commit: git.Commit) -> Optional[str]:
        """Helper method to find the most recent parent measured in number of commits for which there is data."""
        heap: list[tuple[int, git.Commit]] = [(0, commit)]
        seen_hexsha8 = set()
        while heap:
            depth, current_commit = heapq.heappop(heap)
            current_hexsha8 = commit.hexsha[:self.build_len]
            if current_hexsha8 in self.builds:
                return current_hexsha8
            for parent in commit.parents:
                parent_hexsha8 = parent.hexsha[:self.build_len]
                if parent_hexsha8 not in seen_hexsha8:
                    seen_hexsha8.add(parent_hexsha8)
                    heapq.heappush(heap, (depth + 1, parent))
        return None

    def get_all_parent_hexsha8s(self, commit: git.Commit) -> Sequence[str]:
        """Helper method to recursively get hexsha8 values for all parents of a commit."""
        unvisited = [commit]
        visited   = []

        while unvisited:
            current_commit = unvisited.pop(0)
            visited.append(current_commit.hexsha[:self.build_len])
            for parent in current_commit.parents:
                if parent.hexsha[:self.build_len] not in visited:
                    unvisited.append(parent)

        return visited

    def get_commit_name(self, hexsha8: str) -> str:
        """Helper method to find a human-readable name for a commit if possible."""
        if self.repo is None:
            return hexsha8
        for h in self.repo.heads:
            if h.commit.hexsha[:self.build_len] == hexsha8:
                return h.name
        for t in self.repo.tags:
            if t.commit.hexsha[:self.build_len] == hexsha8:
                return t.name
        return hexsha8

    def get_commit_hexsha8(self, name: str) -> Optional[str]:
        """Helper method to search for a commit given a human-readable name."""
        if self.repo is None:
            return None
        for h in self.repo.heads:
            if h.name == name:
                return h.commit.hexsha[:self.build_len]
        for t in self.repo.tags:
            if t.name == name:
                return t.commit.hexsha[:self.build_len]
        for c in self.repo.iter_commits("--all"):
            if c.hexsha[:self.build_len] == name[:self.build_len]:
                return c.hexsha[:self.build_len]
        return None

    def builds_timestamp(self, reverse: bool = False) -> Union[Iterator[tuple], Sequence[tuple]]:
        """Helper method that gets rows of (build_commit, test_time) sorted by the latter."""
        return []

    def get_rows(self, properties: list[str], hexsha8_baseline: str, hexsha8_compare: str) -> Sequence[tuple]:
        """
        Helper method that gets table rows for some list of properties.
        Rows are created by combining those where all provided properties are equal.
        The resulting rows are then grouped by the provided properties and the t/s values are averaged.
        The returned rows are unique in terms of property combinations.
        """
        return []


class LlamaBenchDataSQLite3(LlamaBenchData):
    connection: sqlite3.Connection
    cursor: sqlite3.Cursor

    def __init__(self):
        super().__init__()
        self.connection = sqlite3.connect(":memory:")
        self.cursor = self.connection.cursor()
        self.cursor.execute(f"CREATE TABLE test({', '.join(' '.join(x) for x in zip(DB_FIELDS, DB_TYPES))});")

    def _builds_init(self):
        if self.connection:
            self.build_len_min = self.cursor.execute("SELECT MIN(LENGTH(build_commit)) from test;").fetchone()[0]
            self.build_len_max = self.cursor.execute("SELECT MAX(LENGTH(build_commit)) from test;").fetchone()[0]

            if self.build_len_min != self.build_len_max:
                logger.warning("Data contains commit hashes of differing lengths. It's possible that the wrong commits will be compared. "
                               "Try purging the the database of old commits.")
                self.cursor.execute(f"UPDATE test SET build_commit = SUBSTRING(build_commit, 1, {self.build_len_min});")

            builds = self.cursor.execute("SELECT DISTINCT build_commit FROM test;").fetchall()
            self.builds = list(map(lambda b: b[0], builds))  # list[tuple[str]] -> list[str]
        super()._builds_init()

    def builds_timestamp(self, reverse: bool = False) -> Union[Iterator[tuple], Sequence[tuple]]:
        data = self.cursor.execute(
            "SELECT build_commit, test_time FROM test ORDER BY test_time;").fetchall()
        return reversed(data) if reverse else data

    def get_rows(self, properties: list[str], hexsha8_baseline: str, hexsha8_compare: str) -> Sequence[tuple]:
        select_string = ", ".join(
            [f"tb.{p}" for p in properties] + ["tb.n_prompt", "tb.n_gen", "tb.n_depth", "AVG(tb.avg_ts)", "AVG(tc.avg_ts)"])
        equal_string = " AND ".join(
            [f"tb.{p} = tc.{p}" for p in KEY_PROPERTIES] + [
                f"tb.build_commit = '{hexsha8_baseline}'", f"tc.build_commit = '{hexsha8_compare}'"]
        )
        group_order_string = ", ".join([f"tb.{p}" for p in properties] + ["tb.n_gen", "tb.n_prompt", "tb.n_depth"])
        query = (f"SELECT {select_string} FROM test tb JOIN test tc ON {equal_string} "
                 f"GROUP BY {group_order_string} ORDER BY {group_order_string};")
        return self.cursor.execute(query).fetchall()


class LlamaBenchDataSQLite3File(LlamaBenchDataSQLite3):
    def __init__(self, data_file: str):
        super().__init__()

        self.connection.close()
        self.connection = sqlite3.connect(data_file)
        self.cursor = self.connection.cursor()
        self._builds_init()

    @staticmethod
    def valid_format(data_file: str) -> bool:
        connection = sqlite3.connect(data_file)
        cursor = connection.cursor()

        try:
            if cursor.execute("PRAGMA schema_version;").fetchone()[0] == 0:
                raise sqlite3.DatabaseError("The provided input file does not exist or is empty.")
        except sqlite3.DatabaseError as e:
            logger.debug(f'"{data_file}" is not a valid SQLite3 file.', exc_info=e)
            cursor = None

        connection.close()
        return True if cursor else False


class LlamaBenchDataJSONL(LlamaBenchDataSQLite3):
    def __init__(self, data_file: str):
        super().__init__()

        with open(data_file, "r", encoding="utf-8") as fp:
            for i, line in enumerate(fp):
                parsed = json.loads(line)

                for k in parsed.keys() - set(DB_FIELDS):
                    del parsed[k]

                if (missing_keys := self._check_keys(parsed.keys())):
                    raise RuntimeError(f"Missing required data key(s) at line {i + 1}: {', '.join(missing_keys)}")

                self.cursor.execute(f"INSERT INTO test({', '.join(parsed.keys())}) VALUES({', '.join('?' * len(parsed))});", tuple(parsed.values()))

        self._builds_init()

    @staticmethod
    def valid_format(data_file: str) -> bool:
        try:
            with open(data_file, "r", encoding="utf-8") as fp:
                for line in fp:
                    json.loads(line)
                    break
        except Exception as e:
            logger.debug(f'"{data_file}" is not a valid JSONL file.', exc_info=e)
            return False

        return True


class LlamaBenchDataJSON(LlamaBenchDataSQLite3):
    def __init__(self, data_files: list[str]):
        super().__init__()

        for data_file in data_files:
            with open(data_file, "r", encoding="utf-8") as fp:
                parsed = json.load(fp)

                for i, entry in enumerate(parsed):
                    for k in entry.keys() - set(DB_FIELDS):
                        del entry[k]

                    if (missing_keys := self._check_keys(entry.keys())):
                        raise RuntimeError(f"Missing required data key(s) at entry {i + 1}: {', '.join(missing_keys)}")

                    self.cursor.execute(f"INSERT INTO test({', '.join(entry.keys())}) VALUES({', '.join('?' * len(entry))});", tuple(entry.values()))

        self._builds_init()

    @staticmethod
    def valid_format(data_files: list[str]) -> bool:
        if not data_files:
            return False

        for data_file in data_files:
            try:
                with open(data_file, "r", encoding="utf-8") as fp:
                    json.load(fp)
            except Exception as e:
                logger.debug(f'"{data_file}" is not a valid JSON file.', exc_info=e)
                return False

        return True


class LlamaBenchDataCSV(LlamaBenchDataSQLite3):
    def __init__(self, data_files: list[str]):
        super().__init__()

        for data_file in data_files:
            with open(data_file, "r", encoding="utf-8") as fp:
                for i, parsed in enumerate(csv.DictReader(fp)):
                    keys = set(parsed.keys())

                    for k in keys - set(DB_FIELDS):
                        del parsed[k]

                    if (missing_keys := self._check_keys(keys)):
                        raise RuntimeError(f"Missing required data key(s) at line {i + 1}: {', '.join(missing_keys)}")

                    self.cursor.execute(f"INSERT INTO test({', '.join(parsed.keys())}) VALUES({', '.join('?' * len(parsed))});", tuple(parsed.values()))

        self._builds_init()

    @staticmethod
    def valid_format(data_files: list[str]) -> bool:
        if not data_files:
            return False

        for data_file in data_files:
            try:
                with open(data_file, "r", encoding="utf-8") as fp:
                    for parsed in csv.DictReader(fp):
                        break
            except Exception as e:
                logger.debug(f'"{data_file}" is not a valid CSV file.', exc_info=e)
                return False

        return True


bench_data = None
if len(input_file) == 1:
    if LlamaBenchDataSQLite3File.valid_format(input_file[0]):
        bench_data = LlamaBenchDataSQLite3File(input_file[0])
    elif LlamaBenchDataJSON.valid_format(input_file):
        bench_data = LlamaBenchDataJSON(input_file)
    elif LlamaBenchDataJSONL.valid_format(input_file[0]):
        bench_data = LlamaBenchDataJSONL(input_file[0])
    elif LlamaBenchDataCSV.valid_format(input_file):
        bench_data = LlamaBenchDataCSV(input_file)
else:
    if LlamaBenchDataJSON.valid_format(input_file):
        bench_data = LlamaBenchDataJSON(input_file)
    elif LlamaBenchDataCSV.valid_format(input_file):
        bench_data = LlamaBenchDataCSV(input_file)

if not bench_data:
    raise RuntimeError("No valid (or some invalid) input files found.")

if not bench_data.builds:
    raise RuntimeError(f"{input_file} does not contain any builds.")


hexsha8_baseline = name_baseline = None

# If the user specified a baseline, try to find a commit for it:
if known_args.baseline is not None:
    if known_args.baseline in bench_data.builds:
        hexsha8_baseline = known_args.baseline
    if hexsha8_baseline is None:
        hexsha8_baseline = bench_data.get_commit_hexsha8(known_args.baseline)
        name_baseline = known_args.baseline
    if hexsha8_baseline is None:
        logger.error(f"cannot find data for baseline={known_args.baseline}.")
        sys.exit(1)
# Otherwise, search for the most recent parent of master for which there is data:
elif bench_data.repo is not None:
    hexsha8_baseline = bench_data.find_parent_in_data(bench_data.repo.heads.master.commit)

    if hexsha8_baseline is None:
        logger.error("No baseline was provided and did not find data for any master branch commits.\n")
        parser.print_help()
        sys.exit(1)
else:
    logger.error("No baseline was provided and the current working directory "
                 "is not part of a git repository from which a baseline could be inferred.\n")
    parser.print_help()
    sys.exit(1)


name_baseline = bench_data.get_commit_name(hexsha8_baseline)

hexsha8_compare = name_compare = None

# If the user has specified a compare value, try to find a corresponding commit:
if known_args.compare is not None:
    if known_args.compare in bench_data.builds:
        hexsha8_compare = known_args.compare
    if hexsha8_compare is None:
        hexsha8_compare = bench_data.get_commit_hexsha8(known_args.compare)
        name_compare = known_args.compare
    if hexsha8_compare is None:
        logger.error(f"cannot find data for compare={known_args.compare}.")
        sys.exit(1)
# Otherwise, search for the commit for llama-bench was most recently run
# and that is not a parent of master:
elif bench_data.repo is not None:
    hexsha8s_master = bench_data.get_all_parent_hexsha8s(bench_data.repo.heads.master.commit)
    for (hexsha8, _) in bench_data.builds_timestamp(reverse=True):
        if hexsha8 not in hexsha8s_master:
            hexsha8_compare = hexsha8
            break

    if hexsha8_compare is None:
        logger.error("No compare target was provided and did not find data for any non-master commits.\n")
        parser.print_help()
        sys.exit(1)
else:
    logger.error("No compare target was provided and the current working directory "
                 "is not part of a git repository from which a compare target could be inferred.\n")
    parser.print_help()
    sys.exit(1)

name_compare = bench_data.get_commit_name(hexsha8_compare)

# If the user provided columns to group the results by, use them:
if known_args.show is not None:
    show = known_args.show.split(",")
    unknown_cols = []
    for prop in show:
        if prop not in KEY_PROPERTIES[:-3]:  # Last three values are n_prompt, n_gen, n_depth.
            unknown_cols.append(prop)
    if unknown_cols:
        logger.error(f"Unknown values for --show: {', '.join(unknown_cols)}")
        parser.print_usage()
        sys.exit(1)
    rows_show = bench_data.get_rows(show, hexsha8_baseline, hexsha8_compare)
# Otherwise, select those columns where the values are not all the same:
else:
    rows_full = bench_data.get_rows(KEY_PROPERTIES, hexsha8_baseline, hexsha8_compare)
    properties_different = []
    for i, kp_i in enumerate(KEY_PROPERTIES):
        if kp_i in DEFAULT_SHOW or kp_i in ["n_prompt", "n_gen", "n_depth"]:
            continue
        for row_full in rows_full:
            if row_full[i] != rows_full[0][i]:
                properties_different.append(kp_i)
                break

    show = []
    # Show CPU and/or GPU by default even if the hardware for all results is the same:
    if rows_full and "n_gpu_layers" not in properties_different:
        ngl = int(rows_full[0][KEY_PROPERTIES.index("n_gpu_layers")])

        if ngl != 99 and "cpu_info" not in properties_different:
            show.append("cpu_info")

    show += properties_different

    index_default = 0
    for prop in ["cpu_info", "gpu_info", "n_gpu_layers", "main_gpu"]:
        if prop in show:
            index_default += 1
    show = show[:index_default] + DEFAULT_SHOW + show[index_default:]
    for prop in DEFAULT_HIDE:
        try:
            show.remove(prop)
        except ValueError:
            pass

    # Add plot_x parameter to parameters to show if it's not already present:
    if known_args.plot:
        for k, v in PRETTY_NAMES.items():
            if v == known_args.plot_x and k not in show:
                show.append(k)
                break

    rows_show = bench_data.get_rows(show, hexsha8_baseline, hexsha8_compare)

if not rows_show:
    logger.error(f"No comparable data was found between {name_baseline} and {name_compare}.\n")
    sys.exit(1)

table = []
for row in rows_show:
    n_prompt = int(row[-5])
    n_gen    = int(row[-4])
    n_depth  = int(row[-3])
    if n_prompt != 0 and n_gen == 0:
        test_name = f"pp{n_prompt}"
    elif n_prompt == 0 and n_gen != 0:
        test_name = f"tg{n_gen}"
    else:
        test_name = f"pp{n_prompt}+tg{n_gen}"
    if n_depth != 0:
        test_name = f"{test_name}@d{n_depth}"
    #           Regular columns    test name    avg t/s values              Speedup
    #            VVVVVVVVVVVVV     VVVVVVVVV    VVVVVVVVVVVVVV              VVVVVVV
    table.append(list(row[:-5]) + [test_name] + list(row[-2:]) + [float(row[-1]) / float(row[-2])])

# Some a-posteriori fixes to make the table contents prettier:
for bool_property in BOOL_PROPERTIES:
    if bool_property in show:
        ip = show.index(bool_property)
        for row_table in table:
            row_table[ip] = "Yes" if int(row_table[ip]) == 1 else "No"

if "model_type" in show:
    ip = show.index("model_type")
    for (old, new) in MODEL_SUFFIX_REPLACE.items():
        for row_table in table:
            row_table[ip] = row_table[ip].replace(old, new)

if "model_size" in show:
    ip = show.index("model_size")
    for row_table in table:
        row_table[ip] = float(row_table[ip]) / 1024 ** 3

if "gpu_info" in show:
    ip = show.index("gpu_info")
    for row_table in table:
        for gns in GPU_NAME_STRIP:
            row_table[ip] = row_table[ip].replace(gns, "")

        gpu_names = row_table[ip].split(", ")
        num_gpus = len(gpu_names)
        all_names_the_same = len(set(gpu_names)) == 1
        if len(gpu_names) >= 2 and all_names_the_same:
            row_table[ip] = f"{num_gpus}x {gpu_names[0]}"

headers  = [PRETTY_NAMES[p] for p in show]
headers += ["Test", f"t/s {name_baseline}", f"t/s {name_compare}", "Speedup"]

if known_args.plot:
    def create_performance_plot(table_data: list[list[str]], headers: list[str], baseline_name: str, compare_name: str, output_file: str, plot_x_param: str, log_scale: bool = False):
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError as e:
            logger.error("matplotlib is required for --plot.")
            raise e

        data_headers = headers[:-4] # Exclude the last 4 columns (Test, baseline t/s, compare t/s, Speedup)
        plot_x_index = None
        plot_x_label = plot_x_param

        if plot_x_param not in ["n_prompt", "n_gen", "n_depth"]:
            pretty_name = PRETTY_NAMES.get(plot_x_param, plot_x_param)
            if pretty_name in data_headers:
                plot_x_index = data_headers.index(pretty_name)
                plot_x_label = pretty_name
            elif plot_x_param in data_headers:
                plot_x_index = data_headers.index(plot_x_param)
                plot_x_label = plot_x_param
            else:
                logger.error(f"Parameter '{plot_x_param}' not found in current table columns. Available columns: {', '.join(data_headers)}")
                return

        grouped_data = {}

        for i, row in enumerate(table_data):
            group_key_parts = []
            test_name = row[-4]

            base_test = ""
            x_value = None

            if plot_x_param in ["n_prompt", "n_gen", "n_depth"]:
                for j, val in enumerate(row[:-4]):
                    header_name = data_headers[j]
                    if val is not None and str(val).strip():
                        group_key_parts.append(f"{header_name}={val}")

                if plot_x_param == "n_prompt" and "pp" in test_name:
                    base_test = test_name.split("@")[0]
                    x_value = base_test
                elif plot_x_param == "n_gen" and "tg" in test_name:
                    x_value = test_name.split("@")[0]
                elif plot_x_param == "n_depth" and "@d" in test_name:
                    base_test = test_name.split("@d")[0]
                    x_value = int(test_name.split("@d")[1])
                else:
                    base_test = test_name

                if base_test.strip():
                    group_key_parts.append(f"Test={base_test}")
            else:
                for j, val in enumerate(row[:-4]):
                    if j != plot_x_index:
                        header_name = data_headers[j]
                        if val is not None and str(val).strip():
                            group_key_parts.append(f"{header_name}={val}")
                    else:
                        x_value = val

                group_key_parts.append(f"Test={test_name}")

            group_key = tuple(group_key_parts)

            if group_key not in grouped_data:
                grouped_data[group_key] = []

            grouped_data[group_key].append({
                'x_value': x_value,
                'baseline': float(row[-3]),
                'compare': float(row[-2]),
                'speedup': float(row[-1])
            })

        if not grouped_data:
            logger.error("No data available for plotting")
            return

        def make_axes(num_groups, max_cols=2, base_size=(8, 4)):
            from math import ceil
            cols = 1 if num_groups == 1 else min(max_cols, num_groups)
            rows = ceil(num_groups / cols)

            # Scale figure size by grid dimensions
            w, h = base_size
            fig, ax_arr = plt.subplots(rows, cols,
                                       figsize=(w * cols, h * rows),
                                       squeeze=False)

            axes = ax_arr.flatten()[:num_groups]
            return fig, axes

        num_groups = len(grouped_data)
        fig, axes = make_axes(num_groups)

        plot_idx = 0

        for group_key, points in grouped_data.items():
            if plot_idx >= len(axes):
                break
            ax = axes[plot_idx]

            try:
                points_sorted = sorted(points, key=lambda p: float(p['x_value']) if p['x_value'] is not None else 0)
                x_values = [float(p['x_value']) if p['x_value'] is not None else 0 for p in points_sorted]
            except ValueError:
                points_sorted = sorted(points, key=lambda p: group_key)
                x_values = [p['x_value'] for p in points_sorted]

            baseline_vals = [p['baseline'] for p in points_sorted]
            compare_vals = [p['compare'] for p in points_sorted]

            ax.plot(x_values, baseline_vals, 'o-', color='skyblue',
                    label=f'{baseline_name}', linewidth=2, markersize=6)
            ax.plot(x_values, compare_vals, 's--', color='lightcoral', alpha=0.8,
                    label=f'{compare_name}', linewidth=2, markersize=6)

            if log_scale:
                ax.set_xscale('log', base=2)
                unique_x = sorted(set(x_values))
                ax.set_xticks(unique_x)
                ax.set_xticklabels([str(int(x)) for x in unique_x])

            title_parts = []
            for part in group_key:
                if '=' in part:
                    key, value = part.split('=', 1)
                    title_parts.append(f"{key}: {value}")

            title = ', '.join(title_parts) if title_parts else "Performance comparison"

            ax.set_xlabel(plot_x_label, fontsize=12, fontweight='bold')
            ax.set_ylabel('Tokens per second (t/s)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f'Performance comparison: {compare_name} vs. {baseline_name}',
                     fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=1)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    create_performance_plot(table, headers, name_baseline, name_compare, known_args.plot, known_args.plot_x, known_args.plot_log_scale)

print(tabulate( # noqa: NP100
    table,
    headers=headers,
    floatfmt=".2f",
    tablefmt=known_args.output
))
