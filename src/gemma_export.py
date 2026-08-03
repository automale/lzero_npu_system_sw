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

def parse_operand(arg):
    if hasattr(arg, 'name'): return f"reg_{arg.name}"
    elif isinstance(arg, (list, tuple)): return f"[{', '.join([parse_operand(a) for a in arg])}]"
    else: return str(arg)

# --- 상태 추적용 변수 ---
current_phase = "INIT"  # INIT -> EMBED -> LAYER_0 -> ... -> LM_HEAD
current_sub_block = ""
indent = ""

print("\n// ==========================================")
print("// Gemma-2B NPU Firmware (Block-Structured)")
print("// ==========================================\n")

for node in exported_program.graph.nodes:
    if node.op == "placeholder" or node.op == "get_attr":
        continue # 가중치 로드 부분은 생략하거나 별도 함수로 뺌

    if node.op == "call_function":
        op_name = str(node.target)
        dst = f"reg_{node.name}"
        srcs = [parse_operand(arg) for arg in node.args]
        
        # 1. 현재 실행 중인 Transformer 블록 추론 로직
        # 노드의 입력(srcs)으로 들어가는 가중치 이름을 보고 현재 위치를 파악합니다.
        block_hint = " ".join(srcs)
        
        # 1-1. Embedding 레이어 감지
        if "embed_tokens" in block_hint and current_phase != "EMBED":
            if current_phase != "INIT": print(f"{indent}}}\n")
            current_phase = "EMBED"
            print("// === [Phase 1: Token Embedding & Scaling] ===")
            print("{")
            indent = "    "
            
        # 1-2. Transformer Layer 반복문 감지
        elif "layers_" in block_hint:
            # "layers_0_", "layers_1_" 등에서 숫자 추출
            layer_idx = [s for s in block_hint.split("_") if s.isdigit()][0]
            expected_phase = f"LAYER_{layer_idx}"
            
            # 새로운 레이어가 시작될 때
            if current_phase != expected_phase:
                if current_phase != "EMBED": print("    }\n}\n") # 이전 레이어 닫기
                current_phase = expected_phase
                print(f"// === [Phase 2: Transformer Layer {layer_idx}] ===")
                print("{")
                current_sub_block = ""
            
            # 레이어 내부의 Sub-block (Attention vs MLP) 감지
            if "self_attn" in block_hint or ("rsqrt" in op_name and current_sub_block == ""):
                if current_sub_block != "ATTN":
                    if current_sub_block != "": print("    }")
                    current_sub_block = "ATTN"
                    print("    // --- [Attention Block] ---")
                    print("    {")
            elif "mlp" in block_hint or ("rsqrt" in op_name and current_sub_block == "ATTN"):
                if current_sub_block != "MLP":
                    if current_sub_block != "": print("    }")
                    current_sub_block = "MLP"
                    print("    // --- [MLP Block (GeGLU)] ---")
                    print("    {")
            indent = "        "
            
        # 1-3. Final Layer Norm 및 LM Head 감지
        elif "lm_head" in block_hint or "model_norm" in block_hint:
            if current_phase != "LM_HEAD":
                print("    }\n}\n") # 마지막 레이어 닫기
                current_phase = "LM_HEAD"
                print("// === [Phase 3: Final Norm & LM Head] ===")
                print("{")
                indent = "    "

        # 2. C언어 매크로 출력 생성 (이전과 동일)
        cmd = ""
        if any(k in op_name for k in ["aten.mm", "aten.addmm", "aten.linear", "aten.matmul", "aten.bmm"]):
            src1, src2 = srcs[0] if len(srcs)>0 else "NULL", srcs[1] if len(srcs)>1 else "NULL"
            cmd = f"NPU_GEMM( /*dst*/ {dst}, /*src1*/ {src1}, /*src2*/ {src2} );"
        elif "aten.mul.Tensor" in op_name:
            cmd = f"NPU_VEC_MUL( /*dst*/ {dst}, /*src1*/ {srcs[0]}, /*src2*/ {srcs[1]} );"
        elif "aten.add.Tensor" in op_name:
            cmd = f"NPU_VEC_ADD( /*dst*/ {dst}, /*src1*/ {srcs[0]}, /*src2*/ {srcs[1]} );"
        elif "aten.rsqrt.default" in op_name:
            cmd = f"NPU_RSQRT( /*dst*/ {dst}, /*src*/ {srcs[0]} );"
        elif "aten.gelu" in op_name:
            cmd = f"NPU_GELU( /*dst*/ {dst}, /*src*/ {srcs[0]} );"
        else:
            cmd = f"// [UNMAPPED] {dst} = {op_name}({', '.join(srcs)})"

        if cmd and not cmd.startswith("// [UNMAPPED]"):
            print(f"{indent}{cmd}")

# 파싱 종료 후 마지막 스코프 닫기
print("}")