import torch
import struct
import re
import json
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(
    "google/gemma-2b-it",
    #token=""
)

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
model.eval()
model.config.use_cache = False
dummy_input = torch.randint(0, config.vocab_size, (1, 16), device="meta")
exported_program = torch.export.export(model, (dummy_input,))
nodes = list(exported_program.graph.nodes) # 그래프 노드 리스트

# =========================================================
# [1] 하드웨어 ISA 매핑 (Updated Spec)
# =========================================================
def build_rs1(constant_operand=0, valid_row=0xFFFF, vector_compact_out=0, vector_compact_in=0,
              out_point=0, input_point=0, tile_strided_wr=0, tile_strided_rd=0, transpose_en_wr=0, transpose_en_rd=0,
              rope_en=0, norm_mode=0, act_en=0, alu_mode=0, mxu_en=0, lut_write=0, fusion_en=0, nb_enable=0, cache_enable=0):
    rs1 = 0
    rs1 |= (constant_operand & 0xFFFF) << 0
    rs1 |= (valid_row & 0xFFFF) << 16
    rs1 |= (vector_compact_out & 0x1) << 32
    rs1 |= (vector_compact_in & 0x1) << 33
    rs1 |= (out_point & 0x3) << 34
    rs1 |= (input_point & 0x3) << 36
    rs1 |= (tile_strided_wr & 0x1) << 38
    rs1 |= (tile_strided_rd & 0x3) << 39
    rs1 |= (transpose_en_wr & 0x1) << 41
    rs1 |= (transpose_en_rd & 0x3) << 42
    rs1 |= (rope_en & 0x1) << 44
    rs1 |= (norm_mode & 0x3) << 45
    rs1 |= (act_en & 0x1) << 47
    rs1 |= (alu_mode & 0x3) << 48
    rs1 |= (mxu_en & 0x1) << 50
    rs1 |= (lut_write & 0x1F) << 51
    rs1 |= (fusion_en & 0x1) << 56
    rs1 |= (nb_enable & 0x1) << 57      
    rs1 |= (cache_enable & 0x7) << 58   
    return rs1

def build_rs2_struct(**kwargs):
    fmt = '<12Q6I'
    return struct.pack(
        fmt, kwargs.get('input_addr', 0), kwargs.get('weight1_addr', 0), kwargs.get('weight2_addr', 0),
        kwargs.get('quant_param_addr', 0), kwargs.get('angle_param_addr', 0),
        kwargs.get('act_lut_addr', 0), kwargs.get('exp_lut_addr', 0), kwargs.get('scale_lut_addr', 0),
        kwargs.get('rope_sin_addr', 0), kwargs.get('rope_cos_addr', 0),
        kwargs.get('output_addr', 0), kwargs.get('norm_buff_addr', 0),
        kwargs.get('out_rowNum', 0), kwargs.get('out_intermNum', 0), kwargs.get('out_colNum', 0),
        kwargs.get('input_offset', 0), kwargs.get('weight_offset', 0), kwargs.get('output_offset', 0)
    )

# =========================================================
# [2] Legalizer & Liveness Tracker (DRAM 핑퐁 할당기)
# =========================================================
# 1. 생명주기(Liveness) 사전 분석: 각 텐서가 마지막으로 읽히는 시점 계산
LAST_USAGE = {}
for idx, node in enumerate(nodes):
    LAST_USAGE[node.name] = idx
    for inp in node.all_input_nodes:
        LAST_USAGE[inp.name] = max(LAST_USAGE.get(inp.name, -1), idx)

MEMORY_MAP = {}
FREE_BLOCKS = set()          # 수거된 빈 DRAM 버퍼들 (핑퐁 풀)
CURRENT_DRAM_ADDR = 0x200000 # Workspace 시작 주소
BLOCK_SIZE = 1024 * 1024     # 1MB 할당 단위 (Sequence 길이에 따라 조절 가능)
ACTIVE_ALLOCS = []           # 현재 사용 중인 버퍼들 [(expire_idx, addr)]

SENTINEL_ADDR = 0xFFFFFFFFFFFFFFFF
block_index = 0
relocations = []

exp_lut_loaded = False
act_lut_loaded = False
scale_lut_loaded = False
sin_cos_lut_loaded = False

def get_addr(reg_name, current_idx):
    """가상 레지스터를 DRAM 주소로 변환. 빈 버퍼가 있으면 재사용(핑퐁)함."""
    global CURRENT_DRAM_ADDR
    if reg_name == "NULL": return 0
    
    raw_name = reg_name.replace("reg_", "")
    
    # Weight(가중치)는 재사용 불가 고정 메모리이므로 더미 주소 리턴
    if "weight" in raw_name or "bias" in raw_name or "layernorm" in raw_name:
        return 0x1000000 
        
    if raw_name not in MEMORY_MAP:
        if FREE_BLOCKS:
            # 핑퐁! 이전에 반납된 빈 버퍼를 재사용
            addr = FREE_BLOCKS.pop()
        else:
            # 빈 버퍼가 없으면 새 DRAM 영역 개척
            addr = CURRENT_DRAM_ADDR
            CURRENT_DRAM_ADDR += BLOCK_SIZE
            
        MEMORY_MAP[raw_name] = addr
        # 이 텐서가 수명을 다하는 시점을 기록
        expire_idx = LAST_USAGE.get(raw_name, current_idx)
        ACTIVE_ALLOCS.append((expire_idx, addr))
        
    return MEMORY_MAP[raw_name]

def free_expired_memory(current_idx):
    """현재 시점(current_idx)을 기준으로, 수명이 끝난 버퍼들을 FREE_BLOCKS로 반납."""
    global ACTIVE_ALLOCS
    survivors = []
    for expire_idx, addr in ACTIVE_ALLOCS:
        if current_idx >= expire_idx:
            FREE_BLOCKS.add(addr) # 수거 (재사용 대기열로)
        else:
            survivors.append((expire_idx, addr))
    ACTIVE_ALLOCS = survivors

# =========================================================
# [3] 퓨전 및 변환 헬퍼 함수
# =========================================================
# (이전과 동일)
def resolve_alias_node(node):
    alias_ops = ["aten.view", "aten.reshape", "aten.transpose", "aten.permute", 
                 "aten.clone", "aten.slice", "aten.unsqueeze", "aten._to_copy", "aten.to"]
    curr = node
    while curr is not None and hasattr(curr, 'op') and curr.op == "call_function" and any(op in str(curr.target) for op in alias_ops):
        if len(curr.args) > 0 and hasattr(curr.args[0], 'name'):
            curr = curr.args[0]
        else: break
    return curr

def parse_operand(arg, resolve=False):
    if resolve and hasattr(arg, 'name'): arg = resolve_alias_node(arg)
    if hasattr(arg, 'name'): return f"reg_{arg.name}"
    elif isinstance(arg, (list, tuple)): return f"[{', '.join([parse_operand(a, resolve) for a in arg])}]"
    else: return str(arg)

def is_compute_node(op_name):
    valid_ops = ["aten.mm", "aten.addmm", "aten.linear", "aten.matmul", "aten.bmm", 
                 "aten.mul.Tensor", "aten.add.Tensor", "aten.rsqrt.default", "aten.gelu", "aten.scaled_dot_product_attention"]
    return any(op in op_name for op in valid_ops)

def get_next_npu_node(nodes, current_idx):
    idx = current_idx + 1
    while idx < len(nodes):
        if nodes[idx].op == "call_function" and is_compute_node(str(nodes[idx].target)): return idx, nodes[idx]
        idx += 1
    return -1, None

def get_other_operand(current_node, prev_node):
    if len(current_node.args) < 2: return "NULL"
    arg0_resolved = resolve_alias_node(current_node.args[0])
    arg1_resolved = resolve_alias_node(current_node.args[1])
    if arg0_resolved == prev_node: return parse_operand(current_node.args[1], resolve=True)
    elif arg1_resolved == prev_node: return parse_operand(current_node.args[0], resolve=True)
    return "NULL"

def _is_ancestor(target, node, max_depth=5, _seen=None):
    if _seen is None: _seen = set()
    if max_depth < 0 or node in _seen or not hasattr(node, "name"): return False
    _seen.add(node)
    if node is target: return True
    if not hasattr(node, "all_input_nodes"): return False
    for inp in node.all_input_nodes:
        if _is_ancestor(target, inp, max_depth - 1, _seen): return True
    return False

def is_rmsnorm_at(nodes, idx, base=None):
    if idx < 0 or idx >= len(nodes): return -1, None
    node = nodes[idx]
    if "aten.add.Tensor" in str(node.target) and any(str(arg) == "1e-06" for arg in node.args):
        if base is not None:
            mean_arg = node.args[0] if str(node.args[1]) == "1e-06" else node.args[1]
            if hasattr(mean_arg, 'name') and not _is_ancestor(base, mean_arg): return -1, None
        next_idx, next_node = get_next_npu_node(nodes, idx)
        if next_node and "aten.rsqrt.default" in str(next_node.target): return next_idx, next_node
    return -1, None

def is_gemm(op_name):
    return any(k in op_name for k in ["aten.mm", "aten.addmm", "aten.linear", "aten.matmul", "aten.bmm"])

def try_match_rope(nodes, idx, target_node):
    if idx >= len(nodes) or nodes[idx].op != "call_function": return None
    m0 = nodes[idx]
    if "aten.mul.Tensor" not in str(m0.target): return None
    idx1, m1 = get_next_npu_node(nodes, idx)
    if not m1 or "aten.mul.Tensor" not in str(m1.target): return None
    idx2, add_node = get_next_npu_node(nodes, idx1)
    if not add_node or "aten.add.Tensor" not in str(add_node.target): return None
    if not (_is_ancestor(m0, add_node) and _is_ancestor(m1, add_node)): return None
    if not _is_ancestor(target_node, m0, max_depth=6): return None
    return (idx, idx1, idx2, add_node)

def find_matching_rope(nodes, start_idx, target_node, window=25):
    idx, steps = start_idx, 0
    while idx < len(nodes) and steps < window:
        if nodes[idx].op == "call_function" and "aten.mul.Tensor" in str(nodes[idx].target):
            m = try_match_rope(nodes, idx, target_node)
            if m: return m
        idx += 1
        steps += 1
    return None

# =========================================================
# [4] 메인 컴파일러 엔진 & LUT Loaders
# =========================================================
bin_file = open("compile_to_bin.bin", "wb")
txt_file = open("compile_to_bin.txt", "w")

def emit(macro_str, indent, rs1_val, rs2_kwargs):
    global block_index
    lut_mapping = {
        'act_lut_addr': (0x30, 'ACT_LUT'), 'exp_lut_addr': (0x38, 'EXP_LUT'),
        'scale_lut_addr': (0x40, 'SCALE_LUT'), 'rope_sin_addr': (0x48, 'SIN_LUT'), 'rope_cos_addr': (0x50, 'COS_LUT'),
    }
    for arg_name, (offset, symbol) in lut_mapping.items():
        if rs2_kwargs.get(arg_name) == SENTINEL_ADDR:
            relocations.append({"block_index": block_index, "field_offset": offset, "symbol": symbol})
            
    rs2_packed = build_rs2_struct(**rs2_kwargs)
    if macro_str: txt_file.write(f"{indent}{macro_str}\n")
    bin_file.write(struct.pack('<Q', rs1_val))
    bin_file.write(rs2_packed)
    block_index += 1

def ensure_act_lut(indent):
    global act_lut_loaded
    if not act_lut_loaded:
        txt_file.write(f"{indent}// [ACT(GELU) LUT Setup - 1회 실행]\n")
        emit(f"NPU_LUT_LOAD_ACT( /*act_table*/ SENTINEL );", indent, build_rs1(lut_write=0b10000), {"act_lut_addr": SENTINEL_ADDR})
        act_lut_loaded = True

def ensure_scale_lut(indent):
    global scale_lut_loaded
    if not scale_lut_loaded:
        txt_file.write(f"{indent}// [SCALE LUT Setup - 1회 실행]\n")
        emit(f"NPU_LUT_LOAD_SCALE( /*scale_table*/ SENTINEL );", indent, build_rs1(lut_write=0b00100), {"scale_lut_addr": SENTINEL_ADDR})
        scale_lut_loaded = True

def ensure_exp_lut(indent):
    global exp_lut_loaded
    if not exp_lut_loaded:
        txt_file.write(f"{indent}// [EXP LUT Setup - 1회 실행]\n")
        emit(f"NPU_LUT_LOAD_EXP( /*exp_table*/ SENTINEL );", indent, build_rs1(lut_write=0b01000), {"exp_lut_addr": SENTINEL_ADDR})
        exp_lut_loaded = True

def ensure_sin_cos_lut(indent):
    global sin_cos_lut_loaded
    if not sin_cos_lut_loaded:
        txt_file.write(f"{indent}// [SIN/COS LUT Setup - 1회 실행]\n")
        emit(f"NPU_LUT_LOAD_SIN( /*sin_table*/ SENTINEL );", indent, build_rs1(lut_write=0b00010), {"rope_sin_addr": SENTINEL_ADDR})
        emit(f"NPU_LUT_LOAD_COS( /*cos_table*/ SENTINEL );", indent, build_rs1(lut_write=0b00001), {"rope_cos_addr": SENTINEL_ADDR})
        sin_cos_lut_loaded = True

# =========================================================
# [6] 메인 컴파일 루프 (Prefill)
# =========================================================
i = 0  
skip_idx = set()
current_phase = "INIT"  
current_sub_block = ""

while i < len(nodes):
    # 매 루프 끝마다 수명이 다한 메모리를 반환하여 핑퐁풀(FREE_BLOCKS)에 넣음
    free_expired_memory(i)
    
    if i in skip_idx:
        i += 1; continue
        
    node = nodes[i]
    if node.op != "call_function":
        i += 1; continue

    op_name = str(node.target)
    dst = f"reg_{node.name}"
    srcs_resolved = [parse_operand(arg, resolve=True) for arg in node.args]
    indent = "    "
    
    shape_info = node.meta.get('tensor_meta')
    m, n = 1, 1
    if shape_info:
        dims = shape_info.shape
        m = max(1, dims[0] // 16) if len(dims) > 0 else 1
        n = max(1, dims[-1] // 16) if len(dims) > 0 else 1

    # --- Phase/Block 정렬 추적기 ---
    lookahead_hint = ""
    temp_idx = i
    for _ in range(5):
        temp_idx, next_n = get_next_npu_node(nodes, temp_idx)
        if next_n: lookahead_hint += " " + next_n.name + " " + " ".join([str(a) for a in next_n.args])
        else: break
            
    phase_hint = op_name + " " + node.name + " " + " ".join([parse_operand(a) for a in node.args]) + lookahead_hint
    is_layer = "layers_" in phase_hint
    is_lm_head = ("model_norm" in phase_hint or "lm_head" in phase_hint)
    
    if current_phase == "INIT":
        current_phase = "EMBED"
        txt_file.write("// === [Phase 1: Token Embedding & Scaling] ===\n{\n")
        
    if is_layer:
        match = re.search(r'layers_(\d+)', phase_hint)
        if match:
            layer_idx = match.group(1)
            expected_phase = f"LAYER_{layer_idx}"
            if current_phase != expected_phase:
                if current_phase == "EMBED": txt_file.write("}\n\n")
                elif current_phase.startswith("LAYER"): txt_file.write("    }\n}\n\n")
                current_phase = expected_phase
                txt_file.write(f"       // === [Phase 2: Transformer Layer {layer_idx}] ===\n{{\n")
                current_sub_block = ""
                
            if "input_layernorm" in phase_hint or "self_attn" in phase_hint:
                if current_sub_block != "ATTN":
                    if current_sub_block != "": txt_file.write("    }\n")
                    current_sub_block = "ATTN"
                    txt_file.write("    // --- [Attention Block] ---\n    {\n")
            elif "post_attention_layernorm" in phase_hint or "mlp" in phase_hint:
                if current_sub_block != "MLP":
                    if current_sub_block != "": txt_file.write("    }\n")
                    current_sub_block = "MLP"
                    txt_file.write("    // --- [MLP Block (GeGLU)] ---\n    {\n")
            indent = "        "
    elif is_lm_head and not is_layer and current_phase.startswith("LAYER"):
        if current_phase != "LM_HEAD":
            txt_file.write("    }\n}\n\n")
            current_phase = "LM_HEAD"
            txt_file.write("// === [Phase 3: Final Norm & LM Head] ===\n{\n")

    # =========================================================
    # --- 핵심 퓨전 감지기 (메모리 주소는 get_addr(..., i)로 동적 할당) ---
    # =========================================================
    fusion_matched = False
    
    if is_gemm(op_name):
        src1 = srcs_resolved[0] if len(srcs_resolved) > 0 else "NULL"
        src2 = srcs_resolved[1] if len(srcs_resolved) > 1 else "NULL"
        idx1, node1 = get_next_npu_node(nodes, i)

        if node1 and _is_ancestor(node, node1, max_depth=4):
            op1 = str(node1.target)
            
            if "aten.gelu" in op1:
                ensure_act_lut(indent)
                rs1 = build_rs1(mxu_en=1, act_en=1, input_point=0, out_point=1)
                rs2 = dict(input_addr=get_addr(src1, i), weight1_addr=get_addr(src2, i), output_addr=get_addr(f"reg_{node1.name}", i), out_rowNum=m, out_colNum=n)
                emit(f"NPU_FUSED_GEMM_GELU( /*dst*/ reg_{node1.name}, /*src1*/ {src1}, /*src2*/ {src2} );", indent, rs1, rs2)
                fusion_matched = True; i = idx1 + 1; continue
            
            rms_end_idx, rms_node = is_rmsnorm_at(nodes, idx1, base=node)
            if rms_end_idx != -1:
                ensure_scale_lut(indent)
                rs1 = build_rs1(mxu_en=1, norm_mode=1, input_point=0, out_point=0)
                rs2 = dict(input_addr=get_addr(src1, i), weight1_addr=get_addr(src2, i), output_addr=get_addr(f"reg_{rms_node.name}", i), out_rowNum=m, out_colNum=n)
                emit(f"NPU_FUSED_GEMM_RMSNORM( /*dst*/ reg_{rms_node.name}, /*src1*/ {src1}, /*src2*/ {src2} );", indent, rs1, rs2)
                fusion_matched = True; i = rms_end_idx + 1; continue
            
            if "aten.add.Tensor" in op1 and not any(str(a) == "1e-06" for a in node1.args):
                idx2, node2 = get_next_npu_node(nodes, idx1)
                rms_end_idx, rms_node = is_rmsnorm_at(nodes, idx2, base=node1) if node2 else (-1, None)
                residual = get_other_operand(node1, node)

                if rms_end_idx != -1 and len(node1.users) == 1:
                    ensure_scale_lut(indent)
                    rs1 = build_rs1(mxu_en=1, alu_mode=1, norm_mode=1, input_point=0, out_point=0)
                    rs2 = dict(input_addr=get_addr(src1, i), weight1_addr=get_addr(src2, i), weight2_addr=get_addr(residual, i), output_addr=get_addr(f"reg_{rms_node.name}", i), out_rowNum=m, out_colNum=n)
                    emit(f"NPU_FUSED_GEMM_ADD_RMSNORM( /*dst*/ reg_{rms_node.name}, /*src1*/ {src1}, /*src2*/ {src2}, /*residual*/ {residual} );", indent, rs1, rs2)
                    fusion_matched = True; i = rms_end_idx + 1; continue
                elif residual != "NULL":
                    rs1 = build_rs1(mxu_en=1, alu_mode=1, input_point=0, out_point=1)
                    rs2 = dict(input_addr=get_addr(src1, i), weight1_addr=get_addr(src2, i), weight2_addr=get_addr(residual, i), output_addr=get_addr(f"reg_{node1.name}", i), out_rowNum=m, out_colNum=n)
                    emit(f"NPU_FUSED_GEMM_ADD( /*dst*/ reg_{node1.name}, /*src1*/ {src1}, /*src2*/ {src2}, /*residual*/ {residual} );", indent, rs1, rs2)
                    fusion_matched = True; i = idx1 + 1; continue
            
            if "aten.mul.Tensor" in op1:
                idx2, node2 = get_next_npu_node(nodes, idx1)
                rms_end_idx, rms_node = is_rmsnorm_at(nodes, idx2, base=node1) if node2 else (-1, None)
                factor = get_other_operand(node1, node)
                
                if rms_end_idx != -1 and len(node1.users) == 1:
                    ensure_scale_lut(indent)
                    rs1 = build_rs1(mxu_en=1, alu_mode=2, norm_mode=1, input_point=0, out_point=0)
                    rs2 = dict(input_addr=get_addr(src1, i), weight1_addr=get_addr(src2, i), weight2_addr=get_addr(factor, i), output_addr=get_addr(f"reg_{rms_node.name}", i), out_rowNum=m, out_colNum=n)
                    emit(f"NPU_FUSED_GEMM_MUL_RMSNORM( /*dst*/ reg_{rms_node.name}, /*src1*/ {src1}, /*src2*/ {src2}, /*factor*/ {factor} );", indent, rs1, rs2)
                    fusion_matched = True; i = rms_end_idx + 1; continue
                elif factor != "NULL":
                    rs1 = build_rs1(mxu_en=1, alu_mode=2, input_point=0, out_point=1)
                    rs2 = dict(input_addr=get_addr(src1, i), weight1_addr=get_addr(src2, i), weight2_addr=get_addr(factor, i), output_addr=get_addr(f"reg_{node1.name}", i), out_rowNum=m, out_colNum=n)
                    emit(f"NPU_FUSED_GEMM_MUL( /*dst*/ reg_{node1.name}, /*src1*/ {src1}, /*src2*/ {src2}, /*factor*/ {factor} );", indent, rs1, rs2)
                    fusion_matched = True; i = idx1 + 1; continue

        rope_match = find_matching_rope(nodes, i + 1, node)
        if rope_match and not fusion_matched:
            mul0_idx, mul1_idx, add_idx, add_node = rope_match
            ensure_sin_cos_lut(indent)
            
            rs1_gemm = build_rs1(mxu_en=1, input_point=0, out_point=1)
            rs2_gemm = dict(input_addr=get_addr(src1, i), weight1_addr=get_addr(src2, i), output_addr=get_addr(dst, i), out_rowNum=m, out_colNum=n)
            rs1_rope = build_rs1(rope_en=1, input_point=1, out_point=1)
            rs2_rope = dict(input_addr=get_addr(dst, i), output_addr=get_addr(f"reg_{add_node.name}", i), out_rowNum=m, out_colNum=n)
            
            txt_file.write(f"{indent}NPU_FUSED_GEMM_ROPE( /*dst*/ reg_{add_node.name}, /*src1*/ {src1}, /*src2*/ {src2} );\n")
            emit("", "", rs1_gemm, rs2_gemm)
            emit("", "", rs1_rope, rs2_rope)
            
            skip_idx.update([mul0_idx, mul1_idx, add_idx])
            fusion_matched = True; i += 1; continue

    # =========================================================
    # --- 폴백 연산 & SDPA 분해 ---
    # =========================================================
    if not fusion_matched:
        rms_end_idx, rms_node = is_rmsnorm_at(nodes, i)
        if rms_end_idx != -1:
            ensure_scale_lut(indent)
            src_arg = node.args[0] if str(node.args[1]) == "1e-06" else node.args[1]
            src_res = parse_operand(src_arg, resolve=True)
            rs1 = build_rs1(norm_mode=1, input_point=2, out_point=0)
            rs2 = dict(input_addr=get_addr(src_res, i), output_addr=get_addr(f"reg_{rms_node.name}", i), out_rowNum=m, out_colNum=n)
            emit(f"NPU_RMSNORM( /*dst*/ reg_{rms_node.name}, /*src*/ {src_res} );", indent, rs1, rs2)
            i = rms_end_idx + 1; continue

        cmd = ""
        rs1, rs2 = 0, {}
        if is_gemm(op_name):
            src1 = srcs_resolved[0] if len(srcs_resolved)>0 else "NULL"
            src2 = srcs_resolved[1] if len(srcs_resolved)>1 else "NULL"
            rs1 = build_rs1(mxu_en=1, input_point=0, out_point=1)
            rs2 = dict(input_addr=get_addr(src1, i), weight1_addr=get_addr(src2, i), output_addr=get_addr(dst, i), out_rowNum=m, out_colNum=n)
            cmd = f"NPU_GEMM( /*dst*/ {dst}, /*src1*/ {src1}, /*src2*/ {src2} );"
            
        elif "aten.scaled_dot_product_attention" in op_name:
            ensure_scale_lut(indent)
            ensure_exp_lut(indent)

            q_res = parse_operand(node.args[0], resolve=True) if len(node.args)>0 else "NULL"
            k_res = parse_operand(node.args[1], resolve=True) if len(node.args)>1 else "NULL"
            v_res = parse_operand(node.args[2], resolve=True) if len(node.args)>2 else "NULL"
            
            reg_norm_buff = f"reg_{node.name}_norm_buff"
            reg_softmax = f"reg_{node.name}_softmax"
            
            rs1_p1 = build_rs1(mxu_en=1, transpose_en_rd=2, alu_mode=2, norm_mode=3, nb_enable=1, out_point=0)
            rs2_p1 = dict(input_addr=get_addr(q_res, i), weight1_addr=get_addr(k_res, i), norm_buff_addr=get_addr(reg_norm_buff, i), out_rowNum=m, out_colNum=n)
            emit(f"NPU_SDPA_PHASE1_REDUCE( /*norm_buff*/ {reg_norm_buff}, /*q*/ {q_res}, /*k_trans*/ {k_res} );", indent, rs1_p1, rs2_p1)
            
            rs1_p2a = build_rs1(mxu_en=1, transpose_en_rd=2, norm_mode=3, nb_enable=0, out_point=0)
            rs2_p2a = dict(input_addr=get_addr(q_res, i), weight1_addr=get_addr(k_res, i), norm_buff_addr=get_addr(reg_norm_buff, i), output_addr=get_addr(reg_softmax, i), out_rowNum=m, out_colNum=n)
            emit(f"NPU_SDPA_PHASE2_SOFTMAX( /*dst*/ {reg_softmax}, /*q*/ {q_res}, /*k_trans*/ {k_res}, /*norm_buff*/ {reg_norm_buff} );", indent, rs1_p2a, rs2_p2a)
            
            rs1_p2b = build_rs1(mxu_en=1, input_point=0, out_point=1)
            rs2_p2b = dict(input_addr=get_addr(reg_softmax, i), weight1_addr=get_addr(v_res, i), output_addr=get_addr(dst, i), out_rowNum=m, out_colNum=n)
            emit(f"NPU_BMM_AV( /*dst*/ {dst}, /*attn*/ {reg_softmax}, /*v*/ {v_res} );", indent, rs1_p2b, rs2_p2b)
            
            i += 1; continue
            
        elif "aten.mul.Tensor" in op_name:
            rs1 = build_rs1(alu_mode=2, input_point=1, out_point=1)
            rs2 = dict(input_addr=get_addr(srcs_resolved[0], i), weight1_addr=get_addr(srcs_resolved[1], i), output_addr=get_addr(dst, i), out_rowNum=m, out_colNum=n)
            cmd = f"NPU_VEC_MUL( /*dst*/ {dst}, /*src1*/ {srcs_resolved[0]}, /*src2*/ {srcs_resolved[1]} );"
            
        elif "aten.add.Tensor" in op_name:
            rs1 = build_rs1(alu_mode=1, input_point=1, out_point=1)
            rs2 = dict(input_addr=get_addr(srcs_resolved[0], i), weight1_addr=get_addr(srcs_resolved[1], i), output_addr=get_addr(dst, i), out_rowNum=m, out_colNum=n)
            cmd = f"NPU_VEC_ADD( /*dst*/ {dst}, /*src1*/ {srcs_resolved[0]}, /*src2*/ {srcs_resolved[1]} );"
            
        elif "aten.gelu" in op_name:
            ensure_act_lut(indent)
            rs1 = build_rs1(act_en=1, input_point=1, out_point=1)
            rs2 = dict(input_addr=get_addr(srcs_resolved[0], i), output_addr=get_addr(dst, i), out_rowNum=m, out_colNum=n)
            cmd = f"NPU_GELU( /*dst*/ {dst}, /*src*/ {srcs_resolved[0]} );"

        if cmd:
            emit(cmd, indent, rs1, rs2)

        i += 1 

txt_file.write("}\n")
bin_file.close()
txt_file.close()

with open("gemma_layer0.reloc.json", "w") as f:
    json.dump(relocations, f, indent=4)

print(f"✅ 최종 1-Pass 통합 컴파일 완료! Liveness 기반 DRAM 핑퐁 할당 적용됨.")
print(f"✅ 사용된 최대 Workspace 블록 개수: {(CURRENT_DRAM_ADDR - 0x200000) // BLOCK_SIZE}")