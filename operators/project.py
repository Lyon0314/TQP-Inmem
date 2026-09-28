import torch
import os
from IO.vortex_pipeline import InMemoryPipeline 
from operators.aggregate import evaluate, right_most
from variable import Variable, torch_to_type
from parsing import get_id, get_literals
import constants
from utility.logger import message_logger


def tqp_project(gpu_enable, tensor_group, args, cpu_mem_pool, gpu_mem_pool, name="project"):
  device_name = 'cuda' if gpu_enable else 'cpu'
  preserve_dtype = os.environ.get("TQP_PROJECT_NATIVE", "0") == "1"

  for orig, new in args['rename map'].items():
    tensor_group[new] = tensor_group[orig]

  if len(args['used expr']) > 0:
    output_columns_type = [constants.float_dtype] * len(args['used expr'])
    result_types = []
    need_to_record_types = True

    def project_wrapper(columns):
      nonlocal need_to_record_types
      o_tensors = []
      chunk_rows = next(iter(columns.values())).tensor.size(0)

      for tree in args['used expr']:
        while isinstance(tree, list) and len(tree) == 1:
          tree = tree[0]

        if isinstance(tree, str):
          pass
        else:
          if tree[1] == 'AS' and isinstance(tree[0], str):
            pass
          else:
            result = evaluate(device_name, columns, tree[0] if len(tree) > 1 and tree[1] == 'AS' else tree, chunk_rows, None, None)
            if need_to_record_types:
              result_types.append(result.dtype)

            # malloc_like preserves the actual tensor dtype. Avoid a temporary
            # float conversion followed by a conversion back after the pool copy.
            if not preserve_dtype:
              result = result.to(constants.float_dtype)
            o_tensors.append(result)
      
      need_to_record_types = False
      
      return o_tensors
    
    assert 'mask map' not in args

    pipe = InMemoryPipeline(
      input_columns={k: tensor_group[k] for k in args['materialized in']},
      output_columns_type=output_columns_type,
      operator=project_wrapper,
      gpu_mem_pool=gpu_mem_pool,
      cpu_mem_pool=cpu_mem_pool,
      name=f"{name} projection pipe"
    )
    pipe.do_exchange(20_000_000)
    project_result = pipe.get_result()
    message_logger().debug(project_result)

    for i in range(len(args['materialized out'])):
      # assert project_result[i].tensor.dtype == result_types[i], f'{project_result[i].tensor.dtype}, {result_types[i]}'
      ### TODO: return type as uint8 and convert it here
      tensor_group[args['materialized out'][i]] = Variable(project_result[i].tensor.to(result_types[i]), 
                                                    torch_to_type(result_types[i]),
                                                    project_result[i].memory_ptr,
                                                    project_result[i].placement)

  return -1