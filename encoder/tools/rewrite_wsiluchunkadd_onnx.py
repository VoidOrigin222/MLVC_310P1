#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


PROJECT_DOMAIN = ""


class RewriteError(RuntimeError):
    pass


def tensor_to_array(tensor: TensorProto) -> np.ndarray:
    return numpy_helper.to_array(tensor)


def scalar_value(name: str, initializers: Dict[str, TensorProto]) -> Optional[float]:
    tensor = initializers.get(name)
    if tensor is None:
        return None
    array = tensor_to_array(tensor)
    if array.size != 1:
        return None
    return float(array.reshape(-1)[0])


def int_list_value(name: str, initializers: Dict[str, TensorProto]) -> Optional[List[int]]:
    tensor = initializers.get(name)
    if tensor is None:
        return None
    array = tensor_to_array(tensor)
    if array.size == 0:
        return []
    return [int(v) for v in array.reshape(-1).tolist()]


def value_info_shapes(model: onnx.ModelProto) -> Dict[str, List[int]]:
    inferred = shape_inference.infer_shapes(model)
    out: Dict[str, List[int]] = {}

    def read_value_info(items: Iterable[onnx.ValueInfoProto]) -> None:
        for value in items:
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("shape"):
                continue
            dims: List[int] = []
            ok = True
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                else:
                    ok = False
                    break
            if ok:
                out[value.name] = dims

    read_value_info(inferred.graph.input)
    read_value_info(inferred.graph.value_info)
    read_value_info(inferred.graph.output)
    return out


def value_info_dtypes(model: onnx.ModelProto) -> Dict[str, int]:
    inferred = shape_inference.infer_shapes(model)
    out: Dict[str, int] = {}

    def read_value_info(items: Iterable[onnx.ValueInfoProto]) -> None:
        for value in items:
            tensor_type = value.type.tensor_type
            if tensor_type.elem_type:
                out[value.name] = int(tensor_type.elem_type)

    read_value_info(inferred.graph.input)
    read_value_info(inferred.graph.value_info)
    read_value_info(inferred.graph.output)
    return out


def node_output_map(nodes: Sequence[onnx.NodeProto]) -> Dict[str, onnx.NodeProto]:
    mapping: Dict[str, onnx.NodeProto] = {}
    for node in nodes:
        for output in node.output:
            if output:
                mapping[output] = node
    return mapping


def consumer_map(nodes: Sequence[onnx.NodeProto]) -> Dict[str, List[onnx.NodeProto]]:
    mapping: Dict[str, List[onnx.NodeProto]] = {}
    for node in nodes:
        for value in node.input:
            mapping.setdefault(value, []).append(node)
    return mapping


def get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return int(attr.i)
    return default


def match_slice(
    node: onnx.NodeProto,
    initializers: Dict[str, TensorProto],
) -> Optional[Tuple[int, int, int, int]]:
    if node.op_type != "Slice" or len(node.input) < 3:
        return None
    starts = int_list_value(node.input[1], initializers)
    ends = int_list_value(node.input[2], initializers)
    axes = int_list_value(node.input[3], initializers) if len(node.input) >= 4 and node.input[3] else None
    steps = int_list_value(node.input[4], initializers) if len(node.input) >= 5 and node.input[4] else None
    if starts is None or ends is None:
        return None
    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)
    if len(starts) != 1 or len(ends) != 1 or len(axes) != 1 or len(steps) != 1:
        return None
    return starts[0], ends[0], axes[0], steps[0]


def match_add_pattern(
    add: onnx.NodeProto,
    produced_by: Dict[str, onnx.NodeProto],
    consumes: Dict[str, List[onnx.NodeProto]],
    initializers: Dict[str, TensorProto],
    shapes: Dict[str, List[int]],
    graph_outputs: set,
) -> Optional[Dict[str, object]]:
    if add.op_type != "Add" or len(add.input) != 2 or len(add.output) != 1:
        return None
    if add.output[0] in graph_outputs:
        return None

    slice_nodes = [produced_by.get(add.input[0]), produced_by.get(add.input[1])]
    if any(node is None or node.op_type != "Slice" for node in slice_nodes):
        return None
    slice_a, slice_b = slice_nodes  # type: ignore[assignment]
    if len(slice_a.input) == 0 or len(slice_b.input) == 0:
        return None
    if slice_a.input[0] != slice_b.input[0]:
        return None
    gated_name = slice_a.input[0]
    gated = produced_by.get(gated_name)
    if gated is None or gated.op_type != "Mul" or len(gated.input) != 2:
        return None

    slice_descs = [match_slice(slice_a, initializers), match_slice(slice_b, initializers)]
    if slice_descs[0] is None or slice_descs[1] is None:
        return None
    ordered = sorted(slice_descs, key=lambda x: x[0])  # type: ignore[arg-type]
    (start0, end0, axis0, step0), (start1, end1, axis1, step1) = ordered
    if axis0 != 1 or axis1 != 1 or step0 != 1 or step1 != 1:
        return None
    if start0 != 0 or end0 != start1 or end1 != 2 * end0:
        return None

    sigmoid = None
    source_x = None
    for inp in gated.input:
        maybe = produced_by.get(inp)
        if maybe is not None and maybe.op_type == "Sigmoid":
            sigmoid = maybe
        else:
            source_x = inp
    if sigmoid is None or source_x is None or len(sigmoid.input) != 1:
        return None

    pre_mul = produced_by.get(sigmoid.input[0])
    if pre_mul is None or pre_mul.op_type != "Mul" or len(pre_mul.input) != 2:
        return None

    beta = None
    pre_source = None
    for inp in pre_mul.input:
        value = scalar_value(inp, initializers)
        if value is None:
            pre_source = inp
        else:
            beta = value
    if beta is None or pre_source is None:
        return None
    if pre_source != source_x:
        return None
    if not math.isclose(beta, 4.0, rel_tol=0.0, abs_tol=1e-6):
        return None

    shape = shapes.get(source_x)
    if shape is None or len(shape) != 4:
        return None
    if shape[1] != end1:
        return None

    for intermediate in [
        pre_mul.output[0],
        sigmoid.output[0],
        gated.output[0],
        slice_a.output[0],
        slice_b.output[0],
    ]:
        if intermediate in graph_outputs:
            return None

    if len(consumes.get(slice_a.output[0], [])) != 1 or len(consumes.get(slice_b.output[0], [])) != 1:
        return None
    gated_consumers = consumes.get(gated.output[0], [])
    if {id(node) for node in gated_consumers} != {id(slice_a), id(slice_b)}:
        return None

    return {
        "add": add,
        "source": source_x,
        "gated": gated_name,
        "gated_node": gated,
        "sigmoid_output": sigmoid.output[0],
        "output": add.output[0],
        "nodes": [pre_mul, sigmoid, gated, slice_a, slice_b, add],
        "slice_nodes": [slice_a, slice_b],
        "beta": beta,
        "input_shape": shape,
        "output_shape": [shape[0], shape[1] // 2, shape[2], shape[3]],
    }


def match_post_sigmoid_mul_pattern(
    mul: onnx.NodeProto,
    produced_by: Dict[str, onnx.NodeProto],
    consumes: Dict[str, List[onnx.NodeProto]],
    initializers: Dict[str, TensorProto],
    shapes: Dict[str, List[int]],
    dtypes: Dict[str, int],
    graph_outputs: set,
) -> Optional[Dict[str, object]]:
    if mul.op_type != "Mul" or len(mul.input) != 2 or len(mul.output) != 1:
        return None
    if mul.output[0] in graph_outputs:
        return None

    sigmoid = None
    source_x = None
    for inp in mul.input:
        maybe = produced_by.get(inp)
        if maybe is not None and maybe.op_type == "Sigmoid":
            sigmoid = maybe
        else:
            source_x = inp
    if sigmoid is None or source_x is None or len(sigmoid.input) != 1:
        return None

    pre_mul = produced_by.get(sigmoid.input[0])
    if pre_mul is None or pre_mul.op_type != "Mul" or len(pre_mul.input) != 2:
        return None

    beta = None
    pre_source = None
    for inp in pre_mul.input:
        value = scalar_value(inp, initializers)
        if value is None:
            pre_source = inp
        else:
            beta = value
    if beta is None or pre_source is None or pre_source != source_x:
        return None
    if not math.isclose(beta, 4.0, rel_tol=0.0, abs_tol=1e-6):
        return None

    shape = shapes.get(source_x)
    if shape is None or len(shape) != 4:
        return None
    dtype = dtypes.get(source_x)
    if dtype is not None and dtype not in {TensorProto.FLOAT16, TensorProto.FLOAT}:
        return None

    for intermediate in [pre_mul.output[0], sigmoid.output[0], mul.output[0]]:
        if intermediate in graph_outputs:
            return None
    if len(consumes.get(pre_mul.output[0], [])) != 1 or consumes[pre_mul.output[0]][0] is not sigmoid:
        return None
    if len(consumes.get(sigmoid.output[0], [])) != 1 or consumes[sigmoid.output[0]][0] is not mul:
        return None

    # P1-A intentionally targets the DC branch. The FFN branch feeds two Slice nodes
    # and is already handled by PostSigmoidChunkAdd.
    output_consumers = consumes.get(mul.output[0], [])
    if len(output_consumers) != 1 or output_consumers[0].op_type != "Conv":
        return None

    return {
        "mul": mul,
        "pre_mul_node": pre_mul,
        "sigmoid_node": sigmoid,
        "source": source_x,
        "sigmoid_output": sigmoid.output[0],
        "output": mul.output[0],
        "nodes": [mul],
        "beta": beta,
        "input_shape": shape,
        "output_shape": shape,
    }


def op_type_for_fusion_mode(fusion_mode: str) -> str:
    if fusion_mode == "postsigmoidchunkadd":
        return "PostSigmoidChunkAdd"
    if fusion_mode == "postsigmoidmul":
        return "PostSigmoidMul"
    if fusion_mode == "scaledsilumul":
        return "ScaledSiLUMul"
    raise RewriteError(f"unsupported fusion mode: {fusion_mode}")


def rewrite_model(
    model: onnx.ModelProto,
    fusion_mode: str,
    max_replacements: Optional[int] = None,
    only_match_index: Optional[int] = None,
    postsigmoid_tile_elements: int = 4096,
    postsigmoid_block_dim: int = 8,
    postsigmoid_barrier_mode: int = 0,
    postsigmoidmul_tile_elements: int = 8192,
    postsigmoidmul_block_dim: int = 8,
    scaledsilumul_tile_elements: int = 2048,
    scaledsilumul_block_dim: int = 8,
    scaledsilumul_compute_mode: int = 0,
) -> Tuple[onnx.ModelProto, List[Dict[str, object]]]:
    graph = model.graph
    nodes = list(graph.node)
    initializers = {init.name: init for init in graph.initializer}
    produced_by = node_output_map(nodes)
    consumes = consumer_map(nodes)
    graph_outputs = {out.name for out in graph.output}
    shapes = value_info_shapes(model)
    dtypes = value_info_dtypes(model)

    matches: List[Dict[str, object]] = []
    replaced_node_ids = set()
    replacement_by_node_id = {}

    candidate_index = 0
    for node in nodes:
        if max_replacements is not None and len(matches) >= max_replacements:
            break
        if fusion_mode == "postsigmoidchunkadd":
            match = match_add_pattern(node, produced_by, consumes, initializers, shapes, graph_outputs)
        elif fusion_mode == "postsigmoidmul":
            match = match_post_sigmoid_mul_pattern(
                node, produced_by, consumes, initializers, shapes, dtypes, graph_outputs
            )
        elif fusion_mode == "scaledsilumul":
            match = match_post_sigmoid_mul_pattern(
                node, produced_by, consumes, initializers, shapes, dtypes, graph_outputs
            )
        else:
            raise RewriteError(f"unsupported fusion mode: {fusion_mode}")
        if match is None:
            continue
        if only_match_index is not None and candidate_index != only_match_index:
            candidate_index += 1
            continue
        candidate_index += 1
        if fusion_mode == "postsigmoidchunkadd":
            matched_ids = {id(match["add"]), id(match["gated_node"])}
            matched_ids.update(id(n) for n in match["slice_nodes"])  # type: ignore[index]
            replacement_node = match["add"]
        elif fusion_mode == "postsigmoidmul":
            matched_ids = {id(match["mul"])}
            replacement_node = match["mul"]
        elif fusion_mode == "scaledsilumul":
            matched_ids = {id(match["pre_mul_node"]), id(match["sigmoid_node"]), id(match["mul"])}
            replacement_node = match["mul"]
        else:
            raise RewriteError(f"unsupported fusion mode: {fusion_mode}")
        if replaced_node_ids.intersection(matched_ids):
            raise RewriteError(f"overlapping rewrite match at {node.name}")
        replaced_node_ids.update(matched_ids)
        replacement_by_node_id[id(replacement_node)] = match
        matches.append(match)

    new_nodes: List[onnx.NodeProto] = []
    for node in nodes:
        if id(node) in replacement_by_node_id:
            match = replacement_by_node_id[id(node)]
            op_type = op_type_for_fusion_mode(fusion_mode)
            if fusion_mode == "postsigmoidchunkadd":
                inputs = [match["source"], match["sigmoid_output"]]
                attrs = {
                    "axis": 1,
                    "tile_elements": int(postsigmoid_tile_elements),
                    "block_dim": int(postsigmoid_block_dim),
                    "barrier_mode": int(postsigmoid_barrier_mode),
                }
            elif fusion_mode == "postsigmoidmul":
                inputs = [match["source"], match["sigmoid_output"]]
                attrs = {
                    "tile_elements": int(postsigmoidmul_tile_elements),
                    "block_dim": int(postsigmoidmul_block_dim),
                }
            elif fusion_mode == "scaledsilumul":
                inputs = [match["source"]]
                attrs = {
                    "tile_elements": int(scaledsilumul_tile_elements),
                    "block_dim": int(scaledsilumul_block_dim),
                    "compute_mode": int(scaledsilumul_compute_mode),
                }
            else:
                raise RewriteError(f"unsupported fusion mode: {fusion_mode}")
            custom = helper.make_node(
                op_type,
                inputs=inputs,
                outputs=[match["output"]],
                name=f"{node.name}_{op_type}" if node.name else f"{match['output']}_{op_type}",
                domain=PROJECT_DOMAIN,
                **attrs,
            )
            new_nodes.append(custom)
            continue
        if id(node) in replaced_node_ids:
            continue
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    return model, matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse post-Sigmoid chunk-add ONNX patterns into a custom Ascend op.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument(
        "--fusion-mode",
        choices=["postsigmoidchunkadd", "postsigmoidmul", "scaledsilumul"],
        default="postsigmoidchunkadd",
    )
    parser.add_argument("--max-replacements", type=int, default=None)
    parser.add_argument("--only-match-index", type=int, default=None)
    parser.add_argument(
        "--postsigmoid-tile-elements",
        type=int,
        choices=[2048, 4096, 8192, 16384],
        default=4096,
        help="Tile element count encoded on PostSigmoidChunkAdd nodes for tiling experiments.",
    )
    parser.add_argument(
        "--postsigmoid-block-dim",
        type=int,
        choices=[8, 16, 24, 32],
        default=8,
        help="AICore block count encoded on PostSigmoidChunkAdd nodes for tiling experiments.",
    )
    parser.add_argument(
        "--postsigmoid-barrier-mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="Vector barrier mode for PostSigmoidChunkAdd: 0 keeps both barriers, 1 removes the first barrier.",
    )
    parser.add_argument(
        "--postsigmoidmul-tile-elements",
        type=int,
        choices=[4096, 8192, 16384],
        default=8192,
        help="Tile element count encoded on PostSigmoidMul nodes for tiling experiments.",
    )
    parser.add_argument(
        "--postsigmoidmul-block-dim",
        type=int,
        choices=[8, 16, 24, 32],
        default=8,
        help="AICore block count encoded on PostSigmoidMul nodes for tiling experiments.",
    )
    parser.add_argument(
        "--scaledsilumul-tile-elements",
        type=int,
        choices=[1024, 2048, 4096, 8192],
        default=2048,
        help="Tile element count encoded on ScaledSiLUMul nodes for tiling experiments.",
    )
    parser.add_argument(
        "--scaledsilumul-block-dim",
        type=int,
        choices=[8, 16, 24, 32],
        default=8,
        help="AICore block count encoded on ScaledSiLUMul nodes for tiling experiments.",
    )
    parser.add_argument(
        "--scaledsilumul-compute-mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="ScaledSiLUMul compute mode: 0 computes x * sigmoid(4*x), 1 computes x / (1 + exp(-4*x)).",
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.max_replacements is not None and args.max_replacements < 0:
        raise RewriteError("--max-replacements must be non-negative")
    if args.only_match_index is not None and args.only_match_index < 0:
        raise RewriteError("--only-match-index must be non-negative")
    if args.max_replacements is not None and args.only_match_index is not None:
        raise RewriteError("--max-replacements and --only-match-index cannot be used together")

    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    rewritten, matches = rewrite_model(
        model,
        args.fusion_mode,
        args.max_replacements,
        args.only_match_index,
        args.postsigmoid_tile_elements,
        args.postsigmoid_block_dim,
        args.postsigmoid_barrier_mode,
        args.postsigmoidmul_tile_elements,
        args.postsigmoidmul_block_dim,
        args.scaledsilumul_tile_elements,
        args.scaledsilumul_block_dim,
        args.scaledsilumul_compute_mode,
    )
    if len(matches) != args.expected_count:
        raise RewriteError(f"expected {args.expected_count} matches, found {len(matches)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewritten, args.output)

    report = {
        "source": str(args.model),
        "output": str(args.output),
        "domain": PROJECT_DOMAIN,
        "op_type": op_type_for_fusion_mode(args.fusion_mode),
        "fusion_mode": args.fusion_mode,
        "postsigmoid_tile_elements": (
            args.postsigmoid_tile_elements if args.fusion_mode == "postsigmoidchunkadd" else None
        ),
        "postsigmoid_block_dim": (
            args.postsigmoid_block_dim if args.fusion_mode == "postsigmoidchunkadd" else None
        ),
        "postsigmoid_barrier_mode": (
            args.postsigmoid_barrier_mode if args.fusion_mode == "postsigmoidchunkadd" else None
        ),
        "postsigmoidmul_tile_elements": (
            args.postsigmoidmul_tile_elements if args.fusion_mode == "postsigmoidmul" else None
        ),
        "postsigmoidmul_block_dim": (
            args.postsigmoidmul_block_dim if args.fusion_mode == "postsigmoidmul" else None
        ),
        "scaledsilumul_tile_elements": (
            args.scaledsilumul_tile_elements if args.fusion_mode == "scaledsilumul" else None
        ),
        "scaledsilumul_block_dim": (
            args.scaledsilumul_block_dim if args.fusion_mode == "scaledsilumul" else None
        ),
        "scaledsilumul_compute_mode": (
            args.scaledsilumul_compute_mode if args.fusion_mode == "scaledsilumul" else None
        ),
        "match_count": len(matches),
        "matches": [
            {
                "match_index": index,
                "output": str(match["output"]),
                "source": str(match["source"]),
                "sigmoid_output": str(match["sigmoid_output"]),
                "gated": str(match["gated"]) if "gated" in match else None,
                "custom_inputs": [str(match["source"])]
                if args.fusion_mode == "scaledsilumul"
                else [str(match["source"]), str(match["sigmoid_output"])],
                "beta": float(match["beta"]),
                "tile_elements": {
                    "postsigmoidchunkadd": args.postsigmoid_tile_elements,
                    "postsigmoidmul": args.postsigmoidmul_tile_elements,
                    "scaledsilumul": args.scaledsilumul_tile_elements,
                }[args.fusion_mode],
                "block_dim": {
                    "postsigmoidchunkadd": args.postsigmoid_block_dim,
                    "postsigmoidmul": args.postsigmoidmul_block_dim,
                    "scaledsilumul": args.scaledsilumul_block_dim,
                }[args.fusion_mode],
                "barrier_mode": (
                    args.postsigmoid_barrier_mode if args.fusion_mode == "postsigmoidchunkadd" else None
                ),
                "compute_mode": args.scaledsilumul_compute_mode
                if args.fusion_mode == "scaledsilumul"
                else None,
                "input_shape": match["input_shape"],
                "output_shape": match["output_shape"],
            }
            for index, match in enumerate(matches)
        ],
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(args.model), "output": str(args.output), "match_count": len(matches)}))


if __name__ == "__main__":
    main()
