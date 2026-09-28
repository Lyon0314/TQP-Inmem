import argparse
import time
import json
import torch
import pandas as pd
import pickle
import os
import csv
import sys
# TQP: Parsing Layer
import parsing
# TQP: Tensor Programs
from operators.scan import tqp_scan
from operators.filter import tqp_filter
from operators.project import tqp_project
from operators.aggregate import tqp_hash_aggregate
from operators.sort import tqp_sort
from operators.output import tqp_output
from operators.hashjoin import join_vortex
from utility.logger import datasize_logger, torch_profiler, perf_logger, message_logger, set_torch_profiler, set_datasize_logger, set_perf_logger, set_message_logger
from IO.pinned_mem import PinnedMemory, GPUMemory

subquery_result = None
gpu_enable          = True
force_generate_plan = False

def warmup():
    exec(1, list(range(1, 23)), None, None, True, None, None, None)

pinned_pool = None
gpu_memory_pool = None

def exec(scale_factor, qids, perf_report, log, no_debug, torch_profile_dir, datasize_profile_dir, time_output_file):

    # message_logger().debug("[Read Configuration & Schema JSON]")
    with open('config.json', 'r') as f:
        config_json = json.load(f)
    with open('./queries/schema.json', 'r') as f:
        schema_json = json.load(f)
    global pinned_pool
    global gpu_memory_pool

    print("Pinning CPU memory...", flush=True)
    if scale_factor == 1:
        pinned_pool = PinnedMemory(capacity_gb=15, block_size_mb= int(15 / 2000 * 1000 ))
    elif scale_factor == 10:
        pinned_pool = PinnedMemory(capacity_gb=50, block_size_mb= int(50 / 2000 * 1000 ))
    elif scale_factor == 100:
        pinned_pool = PinnedMemory(capacity_gb=100, block_size_mb= int(100 / 2000 * 1000 ))
    else:
        pinned_pool = PinnedMemory(capacity_gb=250, block_size_mb= int(250 / 2000 * 1000 ))
    print("done!", flush=True)

    if gpu_memory_pool is None:
        print("Allocating GPU Memory...", flush=True)
        gpu_memory_pool = GPUMemory(capacity_gb=8, block_size_mb=(40 / 2000 * 2000))
        print("done!", flush=True)

    set_message_logger(log, not no_debug)

    with open("results.out", "w", encoding="utf-8") as f: # Clear outputs for this run
        pass

    for qid in qids:
        # Read SQL Statement
        set_perf_logger(log, True)
        set_datasize_logger(datasize_profile_dir)
        set_torch_profiler(log, True, os.path.join(torch_profile_dir, str(qid)) if torch_profile_dir else None)

        message_logger().info("======================")
        message_logger().info(f"Start Query {qid}")
        message_logger().info("======================")

        message_logger().info("[Read SQL Statement]")
        sql_path = f'queries/tpch-q{qid}.sql'
        try:
            with open(sql_path, 'r', encoding='utf-8') as file:
                query = file.read()
        except FileNotFoundError:
            message_logger().info(f"{sql_path} Not Found.")

        # Load or generate physical plan
        obj_name = f'./queries/pickles/planobj-q{qid}.pkl'
        if os.path.exists(obj_name) and not force_generate_plan:
            message_logger().info("Plan exists. Loading.")
            with open(obj_name, 'rb') as file:
                query_plans = pickle.load(file)
        else:
            message_logger().info("Plan object does not exist. Generating.")
            # Parse the physical plan into IR graph and detailed description
            message_logger().info("[Spark Initialization]")
            parsing_layer = parsing.ParsingLayer()
            message_logger().info("[Generate Physical Plan]")

            with perf_logger().time("Spark plan generation"):
                query_plans = parsing_layer.generate(query, qid)

            with open(obj_name, 'wb') as file:
                pickle.dump(query_plans, file)

        # Device Info
        message_logger().info("%s %s", "GPU available =", torch.cuda.is_available())
        message_logger().info("%s %s", "GPU count =", torch.cuda.device_count())
        message_logger().info("%s %s", "GPU name =", torch.cuda.get_device_name(0))
        message_logger().info("[CUDA set device]")
        torch.cuda.set_device(0)

        message_logger().info("\n\n\n")

        # [Query Execution]

        # A tensor group refers to a set of related Variables (= tensor + info) produced by an operator.
        # Each operator has its own tensor group as output.

        if gpu_enable:
            torch.cuda.empty_cache()

        perf_logger().start(f"Query {qid}")
        read_time, trans_time, output_time = 0, 0, 0
        q_start_time = time.time()

        subquery_result = None
        for query_plan in query_plans:
            tensor_group = {}
            ops, args, parent_tree = query_plan.ops_idx, query_plan.ops_dict, query_plan.query_parent_map

            if subquery_result and qid == 11:
                subquery_result /= scale_factor

            for u in ops:
                args_u = args[u]
                name = args_u['name']
                datasize_logger().set_operator(name)
                message_logger().info("%s %s", "\n", f"Executing operator {u} {name}")
                num_rows = None

                op_start = time.time()
                if name == 'Scan':
                    with perf_logger().time(f"scan {u}"):
                        time_1 = time.time()
                        num_rows = tqp_scan(
                            gpu_enable,
                            tensor_group,
                            args_u,
                            config_json,
                            schema_json[args_u['table name']],
                            SF=scale_factor,
                            mem_pool=gpu_memory_pool
                        )
                    read_time += time.time() - time_1

                elif name == 'Filter':
                    child = parent_tree[u][0]
                    with torch_profiler().profile(f"filter {u}"):
                        num_rows = tqp_filter(
                            gpu_enable,
                            tensor_group,
                            args_u,
                            subquery_result,
                            cpu_mem_pool=pinned_pool,
                            gpu_mem_pool=gpu_memory_pool,
                            name=f"filter {u}"
                        )

                elif name == 'Project':
                    num_rows = tqp_project(
                        gpu_enable,
                        tensor_group,
                        args_u,
                        gpu_mem_pool=gpu_memory_pool,
                        cpu_mem_pool=pinned_pool,
                        name=f"project {u}"
                    )

                elif name == 'Aggregate':
                    with torch_profiler().profile(f"aggregation {u}"):
                        num_rows = tqp_hash_aggregate(
                            gpu_enable,
                            tensor_group,
                            args_u,
                            cpu_mem_pool=pinned_pool,
                            gpu_mem_pool=gpu_memory_pool,
                            name=f"aggregation {u}"
                        )

                elif name.endswith('Exchange'):
                    assert False, "Exchange is eliminated"

                elif name == 'Sort':
                    num_rows = tqp_sort(
                        gpu_enable,
                        tensor_group,
                        args_u,
                        with_limit=False,
                        name=f"sort {u}"
                    )

                elif name == 'Join':
                    with torch_profiler().profile(f"join {u}"):
                        message_logger().debug(f"[Join Type] {args_u['join type']} JOIN")
                        message_logger().debug(f"[Build] {args_u['build']}")
                        left_keys, right_keys = args_u['left keys'], args_u['right keys']

                        condition_list = []
                        if 'join condition' in args_u.keys():
                            condition_list.append(args_u['join condition'])

                        join_left_key, join_right_key = left_keys[0], right_keys[0]
                        if len(left_keys) == 2:
                            if tensor_group[left_keys[0]].tensor.is_floating_point():
                                assert not tensor_group[left_keys[1]].tensor.is_floating_point(), \
                                    "Do not support join on two float-type keys"

                                join_left_key, join_right_key = left_keys[1], right_keys[1]
                                condition_list.append([f'#{left_keys[0]}', '=', f'#{right_keys[0]}'])
                            else:
                                condition_list.append([f'#{left_keys[1]}', '=', f'#{right_keys[1]}'])

                        message_logger().debug(f"Join on {join_left_key} = {join_right_key}")
                        message_logger().debug(f"condition list = {condition_list}")
                        message_logger().debug(f"[left and right keys], {left_keys}, {right_keys}")

                        join_type = {
                            'Inner': 'inner',
                            'LeftSemi': 'right-semi',
                            'LeftOuter': 'right-outer',
                            'LeftAnti': 'right-anti'
                        }

                        assert args_u['join type'] in join_type.keys(), \
                            f"Unsupport Join Type {args_u['join type']}"

                        num_rows = join_vortex(
                            gpu_enable,
                            tensor_group,
                            args_u,
                            join_left_key,
                            join_right_key,
                            condition_list,
                            join_type[args_u['join type']],
                            cpu_mem_pool=pinned_pool,
                            gpu_mem_pool=gpu_memory_pool,
                            name=f"join {u}",
                        )

                elif name == 'take ordered': # LIMIT
                    num_rows = tqp_sort(
                        gpu_enable,
                        tensor_group,
                        args_u,
                        with_limit=True,
                        name=f"sort lim {u}"
                    )

                elif name == 'AdaptiveSparkPlan': # Treated as output operator
                    with perf_logger().time(f"output {u}"):
                        if qid == 16 and scale_factor >= 100:
                            message_logger().debug("Abandon output.")
                        else:
                            output_time = time.time()
                            message_logger().info(f"Output {u} {args_u['output ids']}")
                            df_result = tqp_output(gpu_enable, tensor_group, args_u)
                            message_logger().info(df_result)
                            output_time = time.time() - output_time

                        if len(query_plans) > 1:
                            subquery_result = float(df_result.iloc[0, 0])
                            message_logger().info(f"subquery_result = %s", subquery_result)

                else:
                    raise Exception(f"Unsupported Operator: {name}")

                if gpu_enable:
                    torch.cuda.synchronize()

                op_end = time.time()
                op_time = op_end - op_start

                message_logger().info(f"Operator Time = {op_time:.4f}")

        perf_logger().stop(f"Query {qid}")
        time_elapsed = time.time() - q_start_time - read_time - output_time   # We manually subtract the time reading from disk and writing to output

        message_logger().info(f"Query {qid} time = {time_elapsed:.6f} seconds")
        if time_output_file:
            file_exists = os.path.isfile(time_output_file)
            with open(time_output_file, "a", newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['SF', 'qid', 'time_elapsed'])
                writer.writerow([scale_factor, qid, f"{time_elapsed:.6f}"])

        pinned_pool.free_all()
        gpu_memory_pool.free_all()

        pd.options.display.float_format = '{:.6f}'.format
        with open("results.out", "a") as f:
            if qid == 16 and scale_factor >= 100:
                f.write("Q16 Output Skipped\n")
            else:
                f.write(df_result.head().to_string())
                f.write("\n")

        if perf_report:
            perf_logger().report(os.path.join(perf_report, f"SF{scale_factor}_{qid}.json"))
        if datasize_profile_dir:
            datasize_logger().report(f"SF{scale_factor}_Q{qid}.json")

if __name__ == "__main__":

    print(torch.__version__)
    print(torch.version.cuda)

    parser = argparse.ArgumentParser()
    parser.add_argument('--SF', type=int, default=1, help='Scale Factor')
    parser.add_argument('--q', type=int, nargs='+', default=list(range(1, 23)), help='List of Query IDs')
    parser.add_argument('--perf_report', type=str, required=False, help='Performance timers directory')
    parser.add_argument("--log", type=str, required=False, help='debug log directory')
    parser.add_argument('--no_debug', action='store_true', help='Disable debug logs')
    parser.add_argument('--torch_profile', type=str, required=False, help='Torch profiling files output directory')
    parser.add_argument('--datasize_profile', type=str, required=False, help='Transferred data size output directory')
    parser.add_argument('--test', action='store_true', help='End to end testing for correcteness on TPCH')
    parser.add_argument('--warmup_iter', type=int, default=0, help='Number of warmup iterations')
    parser.add_argument('--time_output_file', type=str, required=False, help='File to save performance numbers')

    args = parser.parse_args()

    set_message_logger(args.log, False)

    if args.test:
        import pathlib
        import pytest
        test_path = pathlib.Path(__file__).with_name("test_main.py")
        sys.exit(pytest.main(["-q", str(test_path)]))

    message_logger().info("Starting Warmup")
    for _ in range(args.warmup_iter):
        warmup()
    message_logger().info("Warmup Finished")

    with torch.inference_mode():
        exec(args.SF, args.q, args.perf_report, args.log, args.no_debug, args.torch_profile, args.datasize_profile, args.time_output_file)

