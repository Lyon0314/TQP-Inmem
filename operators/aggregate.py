import torch
import os
import constants

import conversion
from variable import Variable, torch_to_type, type_to_torch
from conversion import is_float, str_to_np
from parsing import get_id, right_most
from IO.vortex_pipeline import InMemoryPipeline
from utility.logger import perf_logger, message_logger

def get_literals(tree, literal_set):
  if isinstance(tree, str):
      if '#' in tree and not tree.startswith('Brand#') and tree != '#':
        if 'as' in tree:
          tree = tree.split(' as', 1)[0]
          literal_set.add(get_id(tree))
        else:
          literal_set.add(get_id(tree))
  if isinstance(tree, list):
      for e in tree:
          get_literals(e, literal_set)

def get_result_type(tree, transfer_set, transfer_types):
  if isinstance(tree, str):
    if '#' in tree and not tree.startswith('Brand#') and tree != '#':
      if get_id(tree) in transfer_set and transfer_types[transfer_set.index(get_id(tree))] == constants.float_dtype:
        return constants.float_dtype
      return constants.int_dtype
    if tree.isdigit():
      return constants.int_dtype
    elif is_float(tree):
      return constants.float_dtype
    
  if isinstance(tree, list):
      if tree[0] == 'CASE':
        return get_result_type(tree[4], transfer_set, transfer_types) 
      else:
        for e in tree:
          s = get_result_type(e, transfer_set, transfer_types)
          if s == constants.float_dtype:
            return constants.float_dtype
          if s == torch.int64:
            return torch.int64
        return constants.int_dtype

def get_ops(tree):
  if isinstance(tree, str):
    if tree == 'sum' or tree == 'min' or tree == 'max' or tree == 'count' or tree == 'avg':
      return tree
    return ''
  if isinstance(tree, list):
    for e in tree:
      r = get_ops(e)
      if r != '':
        return r
    return ''

def evaluate(device_name, tensors, tree, cnt_rows, cnt_groups, inverted_index, aggregation_cache=None):
  if isinstance(tree, str):   # date or number
    if tree.isdigit():        # no negative
      return int(tree)
    elif is_float(tree):
      return float(tree)
    elif '#' in tree and tree.split('#')[-1].replace('L', '').isdigit():
      return tensors[get_id(tree)].tensor
    else:
      return tree

  if len(tree) == 1:
    return evaluate(device_name, tensors, tree[0], cnt_rows, cnt_groups, inverted_index, aggregation_cache)

  if len(tree) == 3 and tree[1] == '#':
    assert isinstance(tree[0], list)   # [['avg', ['l_quantity', '#', '313']], '#', '307']
    return evaluate(device_name, tensors, tree[0], cnt_rows, cnt_groups, inverted_index, aggregation_cache)

  if len(tree) == 8:
    assert tree[0] == 'CASE' and tree[1] == 'WHEN' and tree[3] == 'THEN' \
          and tree[5] == 'ELSE' and tree[7] == 'END'
    # e.g., ['CASE', 'WHEN', ['nation#307', '=', 'BRAZIL'], 'THEN', 'volume#306', 'ELSE', '0.0', 'END']
    mask = evaluate(device_name, tensors, tree[2], cnt_rows, cnt_groups, inverted_index, aggregation_cache)
    then = evaluate(device_name, tensors, tree[4], cnt_rows, cnt_groups, inverted_index, aggregation_cache)
    elsc = evaluate(device_name, tensors, tree[6], cnt_rows, cnt_groups, inverted_index, aggregation_cache)

    # DO NOT Support String in CASE
    return torch.where(mask, then, elsc)
  if len(tree) == 2:    # ['avg', ['l_quantity#313']]
    if tree[0] == 'StartsWith':
      # ['StartsWith', ['p_type#348', 'PROMO']]
      id, name = get_id(tree[1][0]), tree[1][1]
      t = tensors[id].tensor
      assert t.shape[-1] >= len(name)
      
      p = torch.tensor(str_to_np(name, len(name)), dtype=constants.string_dtype, device=device_name)
      o = torch.eq(t[:, :len(name)], p).all(dim=1)
      return o
    
    if tree[0] == 'year':
      R = evaluate(device_name, tensors, tree[1], cnt_rows, cnt_groups, inverted_index, aggregation_cache)

      bin_idx = torch.bucketize(R, torch.tensor(conversion.year_edges, device='cuda'), right=True)
      return torch.tensor(conversion.year_range, device='cuda')[bin_idx]

    if tree[0] == 'NOT':
      return ~ evaluate(device_name, tensors, tree[1], cnt_rows, cnt_groups, inverted_index, aggregation_cache)
    
    if tree[0] == 'cast':
      # 'ps_availqty#97 as double'
      literal, _, cast_type = tree[1][0].split(' ')
      if cast_type == 'double':
        # TODO: use double after rocm pytorch fix!!
        return evaluate(device_name, tensors, literal, cnt_rows, cnt_groups, inverted_index, aggregation_cache)
        # return evaluate(device_name, tensors, literal, cnt_rows, cnt_groups, inverted_index, aggregation_cache).double()
      raise ValueError(f"Unsupported cast type: {cast_type}")
    
    if tree[0] == 'substring':
      # e.g., ['substring', ['c_phone#317', '1', '2']]
      assert len(tree[1]) == 3
      sub, l, w = tree[1]
      l, w = int(l) - 1, int(w)
      return evaluate(device_name, tensors, sub, cnt_rows, cnt_groups, inverted_index, aggregation_cache)[:, l:l+w]
    
    if tree[0] not in ['sum', 'avg', 'count', 'min', 'max']:
      raise ValueError(f"Unsupported aggregation function: {tree[0]}")
    
    tensor_tmp = evaluate(device_name, tensors, tree[1], cnt_rows, cnt_groups, inverted_index, aggregation_cache)

    enable_2steps = False
    if cnt_rows / cnt_groups >= (1 << 16): 
      # message_logger().debug(f"{cnt_groups} groups, {cnt_rows} rows, new aggregation.")
      enable_2steps = True
      SUB_GROUPS = max(1, cnt_rows//(cnt_groups << 6))

    def scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, num_subgroups=1024, reduce='max', initv=0):
      N = inverted_index.size(0)
      device = tensor_tmp.device
      # Reuse only within this aggregation invocation, never across queries.
      cache_key = ('flat_index', num_subgroups)
      if aggregation_cache is not None and cache_key in aggregation_cache:
        flat_index = aggregation_cache[cache_key]
      else:
        subgroup_id = torch.arange(N, device=device) % num_subgroups
        flat_index = inverted_index * num_subgroups + subgroup_id
        if aggregation_cache is not None:
          aggregation_cache[cache_key] = flat_index
      result_2d = torch.full((cnt_groups * num_subgroups,), initv, dtype=tensor_tmp.dtype, device=device)
      result_2d = torch.scatter_reduce(result_2d, 0, flat_index, tensor_tmp, reduce=reduce)
      result_2d = result_2d.view(cnt_groups, num_subgroups)
      if reduce == 'max':
        result = result_2d.max(dim=1).values
      elif reduce == 'min':
        result = result_2d.min(dim=1).values
      elif reduce == 'sum':
        result = result_2d.sum(dim=1)
      return result

    if isinstance(tensor_tmp, int):
      tensor_tmp = torch.full((cnt_rows, ), tensor_tmp, dtype=constants.int_dtype, device=device_name)
    elif isinstance(tensor_tmp, float):
      tensor_tmp = torch.full((cnt_rows, ), tensor_tmp, dtype=constants.float_dtype, device=device_name)
    
    if tree[0] == 'min':
      mx = torch.iinfo(tensor_tmp.dtype).max if not tensor_tmp.is_floating_point() else torch.finfo(tensor_tmp.dtype).max
      if enable_2steps:
        result = scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, SUB_GROUPS, 'min', mx)
      else:
        result = torch.full((cnt_groups,), mx, dtype=tensor_tmp.dtype, device=device_name)
        torch.scatter_reduce(result, 0, inverted_index, tensor_tmp, reduce="min", out=result)
    elif tree[0] == 'max':
      # NO Negative
      if enable_2steps:
        result = scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, SUB_GROUPS, 'max', 0)
      else:
        result = torch.zeros(cnt_groups, dtype=tensor_tmp.dtype, device=device_name)
        torch.scatter_reduce(result, 0, inverted_index, tensor_tmp, reduce="max", out=result)
    elif tree[0] == 'avg':
      if enable_2steps:
        result = scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, SUB_GROUPS, 'sum', 0).to(torch.float64)
        tensor_tmp = torch.ones(inverted_index.size(0), dtype=torch.int64, device='cuda')
        tensor_tmp = scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, SUB_GROUPS, 'sum', 0)
        result.div_(tensor_tmp)
      else:
        result = torch.bincount(inverted_index, weights=tensor_tmp, minlength=cnt_groups)
        tensor_tmp = torch.bincount(inverted_index, minlength=cnt_groups)
        result.div_(tensor_tmp)
    else:
      if tensor_tmp.numel() >= 2:
        tensor_tmp.squeeze_()
      elif tensor_tmp.dim() >= 2:
        tensor_tmp.squeeze_(0)

      if tree[0] == 'count':
        tensor_tmp = (tensor_tmp != constants.null).to(constants.int_dtype)

      if enable_2steps:
        result = scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, SUB_GROUPS, 'sum', 0)
      else:
        result = torch.bincount(inverted_index, weights=tensor_tmp, minlength=cnt_groups)

      if not torch.is_floating_point(tensor_tmp):
        result = torch.round(result).to(constants.int_dtype)

    return result

  assert len(tree) == 3

  op = tree[-2]
  
  if op == 'AS':
    return evaluate(device_name, tensors, tree[0], cnt_rows, cnt_groups, inverted_index, aggregation_cache)
  ops = {
    '+': torch.add,
    '-': torch.sub,
    '*': torch.mul,
    '/': torch.div,
    '<=': torch.le,
    '>=': torch.ge,
    '<': torch.lt,
    '>': torch.gt,
    '=': torch.eq,
    '<>': torch.ne,
    'AND': torch.logical_and,
    'OR': torch.logical_or
  }
  if op not in ops:
    raise ValueError(f"Unsupported Operation: {op}")

  L = evaluate(device_name, tensors, tree[0], cnt_rows, cnt_groups, inverted_index, aggregation_cache)
  R = evaluate(device_name, tensors, tree[-1], cnt_rows, cnt_groups, inverted_index, aggregation_cache)

  if isinstance(R, str):    # DO NOT SUPPORT STRING < and >
    if isinstance(L, int):  # 1-URGENT
      assert op == '-'
      return "".join(tree)
    elif len(R) == 1:
      return ops[op](L, ord(R))
    else:
      R = torch.tensor(str_to_np(R, L.shape[-1]), device=device_name)
      return (L == R).all(dim=1)

  return ops[op](L, R)
  

def tqp_hash_aggregate(gpu_enable, tensor_group, args, gpu_mem_pool, cpu_mem_pool, name="aggregation"):
  perf_logger().start(f"{name}")
  device_name = 'cuda' if gpu_enable else 'cpu'

  assert not args['is partial']
  
  transfer_set = args['input ids']
  # assert set(transfer_set) == set(args['input ids']), f"{transfer_set}, {args['input ids']}"
  
  transfer_types = []
  for i in transfer_set:
    transfer_types.extend(tensor_group[i].get_type_torch())
  
  agg_mask_map = {}
  if 'mask map' in args:
    message_logger().debug("Aggregate: accept late materialization")
    for k, v in args['mask map'].items():
      if k in transfer_set:
        agg_mask_map[k] = v
        if v not in transfer_set:
          transfer_set.append(v)
    # transfer_set.extend(list(set(args['mask map'].values())))
    
  message_logger().debug("transfer_set = %s", transfer_set)
  message_logger().debug("transfer_types = %s", transfer_types)

  key_ids, key_ids_new, key_types, key_types_name, key_width = [], [], [], [], []
  agg_ids, agg_types = [], []

  for tree in args['results']:
    while isinstance(tree, list) and len(tree) == 1:
      tree = tree[0]
    if isinstance(tree, str) or (len(tree) == 3 and tree[1] == 'AS' and isinstance(tree[0], str)): # Key
      key_ids.append(get_id(tree if isinstance(tree, str) else tree[0]))
      key_ids_new.append(get_id(tree if isinstance(tree, str) else tree[-1])) # MAY REASSIGN ID
      key_types.extend(tensor_group[key_ids[-1]].get_type_torch())
      key_types_name.append(tensor_group[key_ids[-1]].tensor_type)
      key_width.append(1 if tensor_group[key_ids[-1]].tensor.dim() == 1 else tensor_group[key_ids[-1]].tensor.size(1))
    else:
      agg_types.append(get_result_type(tree, transfer_set, transfer_types))

  for i in range(1, len(key_width)):
    key_width[i] += key_width[i-1]
  
  agg_ids = args['out agg ids']

  def agg_wrapper(columns):
    cnt_rows = next(iter(columns.values())).tensor.size(0)
    message_logger().debug(f"cnt_rows = {cnt_rows}")

    for d in columns.keys():
       message_logger().debug(f"d = {d}, tensor = {columns[d].tensor[0:5]}, \
                              shape = {columns[d].tensor.shape}, type = {columns[d].tensor.dtype}")
    device_out_tensor_group = {}

    if not args['keys']:
      message_logger().debug("No key aggregation")
      # Expected dtype int64 for index
      unique_key_cols, inverted_index = torch.tensor([[1]], device=device_name),\
                                        torch.zeros(cnt_rows, dtype=torch.int64, device=device_name)
      
    else:
      key_ids_d = [get_id(key) for key in args['keys']]
      key_sizes = []

      # Stack keys columns together into a 2D tensor
      key_cols = None

      for key_id in key_ids_d:
        if columns[key_id].tensor.dim() < 2:
          tmp_col = columns[key_id].tensor.unsqueeze(1)
          key_sizes.append(1)
        else:
          tmp_col = columns[key_id].tensor
          key_sizes.append(tmp_col.shape[-1])

        key_cols = tmp_col if key_cols is None else torch.cat((key_cols, tmp_col), dim=1)
        if len(key_sizes) >= 2:
          key_sizes[-1] += key_sizes[-2]

      # New Logic: Hash and then unique
      # enable_hash_aggregate = key_cols.size(1) >= 3
      enable_hash_aggregate = True
      if enable_hash_aggregate:
        hash_coeff = torch.randint(2, 10000000, (key_cols.size(1),), dtype=torch.int64, device='cuda')
        if key_cols.dim() == 1:
          key_cols = key_cols.unsqueeze(1)
        hash_tensor = (key_cols * hash_coeff).sum(dim=1)

        unique_key_cols, inverted_index = torch.unique(hash_tensor, dim=0, return_inverse=True)
        unique_key_cols = torch.zeros((unique_key_cols.size(0), key_cols.size(1)), dtype=key_cols.dtype, device='cuda')
        unique_key_cols.scatter_(0, inverted_index.unsqueeze(1).expand(-1, key_cols.size(1)), key_cols)
      else:
        unique_key_cols, inverted_index = torch.unique(key_cols, dim=0, return_inverse=True)

      del tmp_col, key_cols

    cnt_groups = unique_key_cols.shape[0]
    aggregation_cache = {} if os.environ.get("TQP_AGG_REUSE", "0") == "1" else None

    for tree in args['results']:
      while isinstance(tree, list) and len(tree) == 1:
        tree = tree[0]
      
      agg_id = get_id(right_most(tree))
      if isinstance(tree, str) or (len(tree) == 3 and tree[1] == 'AS' and isinstance(tree[0], str)): # Key
        key_id = get_id(tree if isinstance(tree, str) else tree[0])
        pos = key_ids_d.index(key_id)
        pos_l = 0 if pos == 0 else key_sizes[pos-1]
        pos_r = key_sizes[pos]
        device_out_tensor_group[agg_id] = unique_key_cols[:, pos_l:pos_r]
        device_out_tensor_group[agg_id] = device_out_tensor_group[agg_id].to(columns[key_id].tensor.dtype)
      else:
        # Aggregation
        device_out_tensor_group[agg_id] = evaluate(device_name, columns, tree[0], cnt_rows, cnt_groups, inverted_index, aggregation_cache)
        # e.g, sum((l_extendedprice#53 * (1.0 - l_discount#54)))#319 AS sum_disc_price#307
        # e.g, (0.2 * avg(l_quantity#313)#307) AS (0.2 * avg(l_quantity))#308

    for col in list(columns.keys()):
      del columns[col]
    
    res = []
    for i in key_ids_new:
      res.append(device_out_tensor_group[i])
    for i in agg_ids:
      res.append(device_out_tensor_group[i])
    
    for i in range(len(res)):
      res[i] = res[i].contiguous()
    return res
  
  agg_pipeline = InMemoryPipeline(
    {i: tensor_group[i] for i in transfer_set}, 
    key_types + agg_types,
    agg_wrapper, 
    mask_map=agg_mask_map,
    gpu_mem_pool=gpu_mem_pool,
    cpu_mem_pool=cpu_mem_pool,
    chunk_size=2_000_000_000, 
    name=f"{name} aggregation pipe") 
  
  agg_pipeline.do_exchange(20_000_000)
  result = agg_pipeline.get_result()
  message_logger().debug(result)

  rows_keys = [result[i].tensor for i in range(len(key_types))]
  rows_vals = [result[i].tensor for i in range(len(key_types), len(key_types) + len(agg_types))]
  if len(rows_keys) > 0:
    for i in range(len(rows_vals)):
      tensor_group[agg_ids[i]] = Variable(rows_vals[i], torch_to_type(rows_vals[i].dtype))
    for i in range(len(rows_keys)):
      tensor_group[key_ids_new[i]] = Variable(rows_keys[i],
                                            key_types_name[i])
  else:
    for i in range(len(rows_vals)):
      tensor_group[agg_ids[i]] = Variable(rows_vals[i], torch_to_type(rows_vals[i].dtype))
  
  perf_logger().stop(f"{name}")
  return list(tensor_group.values())[0].tensor.size(0)
 