#!/usr/bin/env python3

import logging
import argparse
import heapq
import sys
import os
from glob import glob
import sqlite3

try:
    import git
    from tabulate import tabulate
except ImportError as e:
    print("the following Python libraries are required: GitPython, tabulate.") # noqa: NP100
    raise e

logger = logging.getLogger("compare-llama-bench")

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

DESCRIPTION = """Creates tables from llama-bench data written to an SQLite database. Example usage (Linux):

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
    "Input SQLite file for comparing commits. "
    "Defaults to 'llama-bench.sqlite' in the current working directory. "
    "If no such file is found and there is exactly one .sqlite file in the current directory, "
    "that file is instead used as input."
)
parser.add_argument("-i", "--input", help=help_i)
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
    f"Legal values: {', '.join(KEY_PROPERTIES[:-2])}. "
    "Defaults to model name (model_type) and CPU and/or GPU name (cpu_info, gpu_info) "
    "plus any column where not all data points are the same. "
    "If the columns are manually specified, then the results for each unique combination of the "
    "specified values are averaged WITHOUT weighing by the --repetitions parameter of llama-bench."
)
parser.add_argument("--check", action="store_true", help="check if all required Python libraries are installed")
parser.add_argument("-s", "--show", help=help_s)
parser.add_argument("--verbose", action="store_true", help="increase output verbosity")

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
if input_file is None and os.path.exists("./llama-bench.sqlite"):
    input_file = "llama-bench.sqlite"
if input_file is None:
    sqlite_files = glob("*.sqlite")
    if len(sqlite_files) == 1:
        input_file = sqlite_files[0]

if input_file is None:
    logger.error("Cannot find a suitable input file, please provide one.\n")
    parser.print_help()
    sys.exit(1)

connection = sqlite3.connect(input_file)
cursor = connection.cursor()

build_len_min: int = cursor.execute("SELECT MIN(LENGTH(build_commit)) from test;").fetchone()[0]
build_len_max: int = cursor.execute("SELECT MAX(LENGTH(build_commit)) from test;").fetchone()[0]

if build_len_min != build_len_max:
    logger.warning(f"{input_file} contains commit hashes of differing lengths. It's possible that the wrong commits will be compared. "
                   "Try purging the the database of old commits.")
    cursor.execute(f"UPDATE test SET build_commit = SUBSTRING(build_commit, 1, {build_len_min});")

build_len: int = build_len_min

builds = cursor.execute("SELECT DISTINCT build_commit FROM test;").fetchall()
builds = list(map(lambda b: b[0], builds))  # list[tuple[str]] -> list[str]

if not builds:
    raise RuntimeError(f"{input_file} does not contain any builds.")

try:
    repo = git.Repo(".", search_parent_directories=True)
except git.InvalidGitRepositoryError:
    repo = None


def find_parent_in_data(commit: git.Commit):
    """Helper function to find the most recent parent measured in number of commits for which there is data."""
    heap: list[tuple[int, git.Commit]] = [(0, commit)]
    seen_hexsha8 = set()
    while heap:
        depth, current_commit = heapq.heappop(heap)
        current_hexsha8 = commit.hexsha[:build_len]
        if current_hexsha8 in builds:
            return current_hexsha8
        for parent in commit.parents:
            parent_hexsha8 = parent.hexsha[:build_len]
            if parent_hexsha8 not in seen_hexsha8:
                seen_hexsha8.add(parent_hexsha8)
                heapq.heappush(heap, (depth + 1, parent))
    return None


def get_all_parent_hexsha8s(commit: git.Commit):
    """Helper function to recursively get hexsha8 values for all parents of a commit."""
    unvisited = [commit]
    visited   = []

    while unvisited:
        current_commit = unvisited.pop(0)
        visited.append(current_commit.hexsha[:build_len])
        for parent in current_commit.parents:
            if parent.hexsha[:build_len] not in visited:
                unvisited.append(parent)

    return visited


def get_commit_name(hexsha8: str):
    """Helper function to find a human-readable name for a commit if possible."""
    if repo is None:
        return hexsha8
    for h in repo.heads:
        if h.commit.hexsha[:build_len] == hexsha8:
            return h.name
    for t in repo.tags:
        if t.commit.hexsha[:build_len] == hexsha8:
            return t.name
    return hexsha8


def get_commit_hexsha8(name: str):
    """Helper function to search for a commit given a human-readable name."""
    if repo is None:
        return None
    for h in repo.heads:
        if h.name == name:
            return h.commit.hexsha[:build_len]
    for t in repo.tags:
        if t.name == name:
            return t.commit.hexsha[:build_len]
    for c in repo.iter_commits("--all"):
        if c.hexsha[:build_len] == name[:build_len]:
            return c.hexsha[:build_len]
    return None


hexsha8_baseline = name_baseline = None

# If the user specified a baseline, try to find a commit for it:
if known_args.baseline is not None:
    if known_args.baseline in builds:
        hexsha8_baseline = known_args.baseline
    if hexsha8_baseline is None:
        hexsha8_baseline = get_commit_hexsha8(known_args.baseline)
        name_baseline = known_args.baseline
    if hexsha8_baseline is None:
        logger.error(f"cannot find data for baseline={known_args.baseline}.")
        sys.exit(1)
# Otherwise, search for the most recent parent of master for which there is data:
elif repo is not None:
    hexsha8_baseline = find_parent_in_data(repo.heads.master.commit)

    if hexsha8_baseline is None:
        logger.error("No baseline was provided and did not find data for any master branch commits.\n")
        parser.print_help()
        sys.exit(1)
else:
    logger.error("No baseline was provided and the current working directory "
                 "is not part of a git repository from which a baseline could be inferred.\n")
    parser.print_help()
    sys.exit(1)


name_baseline = get_commit_name(hexsha8_baseline)

hexsha8_compare = name_compare = None

# If the user has specified a compare value, try to find a corresponding commit:
if known_args.compare is not None:
    if known_args.compare in builds:
        hexsha8_compare = known_args.compare
    if hexsha8_compare is None:
        hexsha8_compare = get_commit_hexsha8(known_args.compare)
        name_compare = known_args.compare
    if hexsha8_compare is None:
        logger.error(f"cannot find data for compare={known_args.compare}.")
        sys.exit(1)
# Otherwise, search for the commit for llama-bench was most recently run
# and that is not a parent of master:
elif repo is not None:
    hexsha8s_master = get_all_parent_hexsha8s(repo.heads.master.commit)
    builds_timestamp = cursor.execute(
        "SELECT build_commit, test_time FROM test ORDER BY test_time;").fetchall()
    for (hexsha8, _) in reversed(builds_timestamp):
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

name_compare = get_commit_name(hexsha8_compare)


def get_rows(properties):
    """
    Helper function that gets table rows for some list of properties.
    Rows are created by combining those where all provided properties are equal.
    The resulting rows are then grouped by the provided properties and the t/s values are averaged.
    The returned rows are unique in terms of property combinations.
    """
    select_string = ", ".join(
        [f"tb.{p}" for p in properties] + ["tb.n_prompt", "tb.n_gen", "tb.n_depth", "AVG(tb.avg_ts)", "AVG(tc.avg_ts)"])
    equal_string = " AND ".join(
        [f"tb.{p} = tc.{p}" for p in KEY_PROPERTIES] + [
            f"tb.build_commit = '{hexsha8_baseline}'", f"tc.build_commit = '{hexsha8_compare}'"]
    )
    group_order_string = ", ".join([f"tb.{p}" for p in properties] + ["tb.n_gen", "tb.n_prompt", "tb.n_depth"])
    query = (f"SELECT {select_string} FROM test tb JOIN test tc ON {equal_string} "
             f"GROUP BY {group_order_string} ORDER BY {group_order_string};")
    return cursor.execute(query).fetchall()


# If the user provided columns to group the results by, use them:
if known_args.show is not None:
    show = known_args.show.split(",")
    unknown_cols = []
    for prop in show:
        if prop not in KEY_PROPERTIES[:-2]:  # Last two values are n_prompt, n_gen.
            unknown_cols.append(prop)
    if unknown_cols:
        logger.error(f"Unknown values for --show: {', '.join(unknown_cols)}")
        parser.print_usage()
        sys.exit(1)
    rows_show = get_rows(show)
# Otherwise, select those columns where the values are not all the same:
else:
    rows_full = get_rows(KEY_PROPERTIES)
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
    rows_show = get_rows(show)

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

print(tabulate( # noqa: NP100
    table,
    headers=headers,
    floatfmt=".2f",
    tablefmt=known_args.output
))
