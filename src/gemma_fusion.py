import torch
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(
    "google/gemma-2b-it",
    # token=""
)

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
model.eval()
model.config.use_cache = False
dummy_input = torch.randint(0, config.vocab_size, (1, 16), device="meta")
exported_program = torch.export.export(model, (dummy_input,))

# ==========================================
# --- Alias 추적 및 피연산자 파싱 ---
# ==========================================
def resolve_alias_node(node):
    # [수정됨] aten._to_copy, aten.to 등 타입 캐스팅 연산자 추가
    alias_ops = ["aten.view", "aten.reshape", "aten.transpose", "aten.permute", 
                 "aten.clone", "aten.slice", "aten.unsqueeze", "aten._to_copy", "aten.to"]
    curr = node
    while curr is not None and hasattr(curr, 'op') and curr.op == "call_function" and any(op in str(curr.target) for op in alias_ops):
        if len(curr.args) > 0 and hasattr(curr.args[0], 'name'):
            curr = curr.args[0]
        else:
            break
    return curr

def parse_operand(arg, resolve=False):
    if resolve and hasattr(arg, 'name'):
        arg = resolve_alias_node(arg)
    if hasattr(arg, 'name'): return f"reg_{arg.name}"
    elif isinstance(arg, (list, tuple)): return f"[{', '.join([parse_operand(a, resolve) for a in arg])}]"
    else: return str(arg)

# ==========================================
# --- 퓨전 엔진용 헬퍼 함수 ---
# ==========================================
def is_compute_node(op_name):
    valid_ops = ["aten.mm", "aten.addmm", "aten.linear", "aten.matmul", "aten.bmm", 
                 "aten.mul.Tensor", "aten.add.Tensor", "aten.rsqrt.default", "aten.gelu",
                 "aten.scaled_dot_product_attention"]
    return any(op in op_name for op in valid_ops)

def get_next_npu_node(nodes, current_idx):
    idx = current_idx + 1
    while idx < len(nodes):
        if nodes[idx].op == "call_function" and is_compute_node(str(nodes[idx].target)):
            return idx, nodes[idx]
        idx += 1
    return -1, None

def get_other_operand(current_node, prev_node):
    if len(current_node.args) < 2: return "NULL"
    
    # prev_node가 직접 1-hop이 아닐 수 있으므로 alias 해상 후 검사
    arg0_resolved = resolve_alias_node(current_node.args[0])
    arg1_resolved = resolve_alias_node(current_node.args[1])
    
    if arg0_resolved == prev_node:
        return parse_operand(current_node.args[1], resolve=True)
    elif arg1_resolved == prev_node:
        return parse_operand(current_node.args[0], resolve=True)
    else:
        # 안전한 Fallback
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
            if hasattr(mean_arg, 'name') and not _is_ancestor(base, mean_arg):
                return -1, None
        next_idx, next_node = get_next_npu_node(nodes, idx)
        if next_node and "aten.rsqrt.default" in str(next_node.target):
            return next_idx, next_node
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

# ==========================================
# --- 메인 컴파일러 프론트엔드 로직 ---
# ==========================================
current_phase = "INIT"  
current_sub_block = ""
indent = ""

nodes = list(exported_program.graph.nodes)
i = 0  
skip_idx = set()

with open("gemma2b_npu_firmware_fused_v7.txt", "w") as f:
    f.write("\n// ==========================================\n")
    f.write("// Gemma-2B NPU Firmware (v7 - Final Ultimate Fusion)\n")
    f.write("// ==========================================\n\n")

    while i < len(nodes):
        if i in skip_idx:
            i += 1
            continue
        node = nodes[i]
        if node.op != "call_function":
            i += 1
            continue

        op_name = str(node.target)
        dst = f"reg_{node.name}"
        srcs_resolved = [parse_operand(arg, resolve=True) for arg in node.args]
        
        # --- 1. 상태 머신 기반 블록 추적 (Phase Fix 유지) ---
        phase_hint = op_name + " " + node.name + " " + " ".join([parse_operand(a) for a in node.args])
        
        is_layer = "layers_" in phase_hint
        is_lm_head = ("lm_head" in phase_hint or "model_norm" in phase_hint)
        
        if current_phase == "INIT":
            current_phase = "EMBED"
            f.write("// === [Phase 1: Token Embedding & Scaling] ===\n{\n")
            indent = "    "
            
        if is_layer:
            layer_idx = [s for s in phase_hint.split("_") if s.isdigit()][0]
            expected_phase = f"LAYER_{layer_idx}"
            if current_phase != expected_phase:
                if current_phase == "EMBED": f.write("}\n\n")
                elif current_phase.startswith("LAYER"): f.write("    }\n}\n\n")
                
                current_phase = expected_phase
                f.write(f"       // === [Phase 2: Transformer Layer {layer_idx}] ===\n{{\n")
                current_sub_block = ""
                
            if "self_attn" in phase_hint or ("rsqrt" in op_name and current_sub_block == ""):
                if current_sub_block != "ATTN":
                    if current_sub_block != "": f.write("    }\n")
                    current_sub_block = "ATTN"
                    f.write("    // --- [Attention Block] ---\n    {\n")
                    indent = "        "
            elif "mlp" in phase_hint or ("rsqrt" in op_name and current_sub_block == "ATTN"):
                if current_sub_block != "MLP":
                    if current_sub_block != "": f.write("    }\n")
                    current_sub_block = "MLP"
                    f.write("    // --- [MLP Block (GeGLU)] ---\n    {\n")
                    indent = "        "
                    
        elif is_lm_head and not is_layer and current_phase.startswith("LAYER"):
            if current_phase != "LM_HEAD":
                f.write("    }\n}\n\n")
                current_phase = "LM_HEAD"
                f.write("// === [Phase 3: Final Norm & LM Head] ===\n{\n")
                indent = "    "

        # --- 2. 슈퍼 매크로 퓨전 감지기 ---
        if is_gemm(op_name):
            src1 = srcs_resolved[0] if len(srcs_resolved) > 0 else "NULL"
            src2 = srcs_resolved[1] if len(srcs_resolved) > 1 else "NULL"
            
            idx1, node1 = get_next_npu_node(nodes, i)

            # [핵심 수정] consumes(1-hop) 대신 _is_ancestor(multi-hop) 적용
            if node1 and _is_ancestor(node, node1, max_depth=4):
                op1 = str(node1.target)
                
                # Case 2-1: GEMM -> GELU
                if "aten.gelu" in op1:
                    f.write(f"{indent}NPU_FUSED_GEMM_GELU( /*dst*/ reg_{node1.name}, /*src1*/ {src1}, /*src2*/ {src2} );\n")
                    i = idx1 + 1; continue
                
                # Case 2-2: GEMM -> RMSNORM
                rms_end_idx, rms_node = is_rmsnorm_at(nodes, idx1, base=node)
                if rms_end_idx != -1:
                    f.write(f"{indent}NPU_FUSED_GEMM_RMSNORM( /*dst*/ reg_{rms_node.name}, /*src1*/ {src1}, /*src2*/ {src2} );\n")
                    i = rms_end_idx + 1; continue
                
                # Case 2-3: GEMM -> ADD (Residual)
                if "aten.add.Tensor" in op1 and not any(str(a) == "1e-06" for a in node1.args):
                    idx2, node2 = get_next_npu_node(nodes, idx1)
                    rms_end_idx, rms_node = is_rmsnorm_at(nodes, idx2, base=node1) if node2 else (-1, None)
                    residual = get_other_operand(node1, node)

                    if rms_end_idx != -1 and len(node1.users) == 1:
                        f.write(f"{indent}NPU_FUSED_GEMM_ADD_RMSNORM( /*dst*/ reg_{rms_node.name}, /*src1*/ {src1}, /*src2*/ {src2}, /*residual*/ {residual} );\n")
                        i = rms_end_idx + 1; continue
                    elif residual != "NULL":
                        f.write(f"{indent}NPU_FUSED_GEMM_ADD( /*dst*/ reg_{node1.name}, /*src1*/ {src1}, /*src2*/ {src2}, /*residual*/ {residual} );\n")
                        i = idx1 + 1; continue
                
                # [복구 및 퓨전 완성] Case 2-4: GEMM -> MUL (up_proj 퓨전)
                if "aten.mul.Tensor" in op1:
                    idx2, node2 = get_next_npu_node(nodes, idx1)
                    rms_end_idx, rms_node = is_rmsnorm_at(nodes, idx2, base=node1) if node2 else (-1, None)
                    factor = get_other_operand(node1, node)
                    
                    if rms_end_idx != -1 and len(node1.users) == 1:
                        f.write(f"{indent}NPU_FUSED_GEMM_MUL_RMSNORM( /*dst*/ reg_{rms_node.name}, /*src1*/ {src1}, /*src2*/ {src2}, /*factor*/ {factor} );\n")
                        i = rms_end_idx + 1; continue
                    elif factor != "NULL":
                        f.write(f"{indent}NPU_FUSED_GEMM_MUL( /*dst*/ reg_{node1.name}, /*src1*/ {src1}, /*src2*/ {src2}, /*factor*/ {factor} );\n")
                        i = idx1 + 1; continue

            # Case 2-5: GEMM -> RoPE
            rope_match = find_matching_rope(nodes, i + 1, node)
            if rope_match:
                mul0_idx, mul1_idx, add_idx, add_node = rope_match
                f.write(f"{indent}NPU_FUSED_GEMM_ROPE( /*dst*/ reg_{add_node.name}, /*src1*/ {src1}, /*src2*/ {src2} );\n")
                skip_idx.update([mul0_idx, mul1_idx, add_idx])
                i += 1; continue

        # --- 3. 단일 명령어 및 특수 매크로 ---
        rms_end_idx, rms_node = is_rmsnorm_at(nodes, i)
        if rms_end_idx != -1:
            src_arg = node.args[0] if str(node.args[1]) == "1e-06" else node.args[1]
            f.write(f"{indent}NPU_RMSNORM( /*dst*/ reg_{rms_node.name}, /*src*/ {parse_operand(src_arg, resolve=True)} );\n")
            i = rms_end_idx + 1
            continue

        cmd = ""
        if is_gemm(op_name):
            src1 = srcs_resolved[0] if len(srcs_resolved)>0 else "NULL"
            src2 = srcs_resolved[1] if len(srcs_resolved)>1 else "NULL"
            cmd = f"NPU_GEMM( /*dst*/ {dst}, /*src1*/ {src1}, /*src2*/ {src2} );"
            
        elif "aten.scaled_dot_product_attention" in op_name:
            q_resolved = parse_operand(node.args[0], resolve=True) if len(node.args)>0 else "NULL"
            k_resolved = parse_operand(node.args[1], resolve=True) if len(node.args)>1 else "NULL"
            v_resolved = parse_operand(node.args[2], resolve=True) if len(node.args)>2 else "NULL"
            
            reg_norm_buff = f"reg_{node.name}_norm_buff"
            
            f.write(f"{indent}// [Online Softmax: 2-Phase Tile FSM (using norm_buff)]\n")
            f.write(f"{indent}NPU_SDPA_PHASE1_REDUCE( /*norm_buff*/ {reg_norm_buff}, /*q*/ {q_resolved}, /*k_trans*/ {k_resolved} );\n")
            f.write(f"{indent}NPU_SDPA_PHASE2_UPDATE( /*dst*/ {dst}, /*q*/ {q_resolved}, /*k_trans*/ {k_resolved}, /*v*/ {v_resolved}, /*norm_buff*/ {reg_norm_buff} );\n")
            i += 1; continue
            
        elif "aten.mul.Tensor" in op_name:
            cmd = f"NPU_VEC_MUL( /*dst*/ {dst}, /*src1*/ {srcs_resolved[0]}, /*src2*/ {srcs_resolved[1]} );"
        elif "aten.add.Tensor" in op_name:
            cmd = f"NPU_VEC_ADD( /*dst*/ {dst}, /*src1*/ {srcs_resolved[0]}, /*src2*/ {srcs_resolved[1]} );"
        elif "aten.gelu" in op_name:
            cmd = f"NPU_GELU( /*dst*/ {dst}, /*src*/ {srcs_resolved[0]} );"

        if cmd:
            f.write(f"{indent}{cmd}\n")

        i += 1 

    f.write("}\n")
