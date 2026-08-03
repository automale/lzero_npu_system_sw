import torch
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(
    "google/gemma-2b-it",
    # token=""
)

with torch.device("meta"): # 메모리 절약을 위해 meta 디바이스 사용
    model = AutoModelForCausalLM.from_config(config)

model.eval()
model.config.use_cache = False
dummy_input = torch.randint(0, config.vocab_size, (1, 16), device="meta")
exported_program = torch.export.export(model, (dummy_input,))

# --- NPU 피연산자 파싱 헬퍼 함수 ---
def parse_operand(arg):
    """
    node.args 안에 들어있는 피연산자들을 NPU가 읽기 편한 문자열로 변환합니다.
    (이전 노드의 결과물인 경우 node.name을 반환, 상수인 경우 값을 반환)
    """
    if hasattr(arg, 'name'):
        return f"reg_{arg.name}" # 레지스터나 메모리 주소 이름으로 치환
    elif isinstance(arg, (list, tuple)):
        return f"[{', '.join([parse_operand(a) for a in arg])}]"
    else:
        return str(arg) # 1e-06 같은 스칼라 상수값

print("\n=== NPU 명령어 매핑 (Operand 명시 버전) ===")
for node in exported_program.graph.nodes:
    
    # [1] 연산 결과가 저장될 Destination 주소 (출력)
    dst = f"reg_{node.name}"
    
    # [2] 연산에 사용될 Source 피연산자 추출 (입력)
    # node.args는 (입력1, 입력2, ...) 형태의 튜플입니다.
    srcs = [parse_operand(arg) for arg in node.args]
    
    if node.op == "get_attr":
        print(f"{dst} = MEM_LOAD( weight_name: {node.target} );")
        
    elif node.op == "placeholder":
        print(f"{dst} = IO_INPUT( {node.name} );")
        
    elif node.op == "call_function":
        op_name = str(node.target)
        
        # 1. GEMM (행렬곱) 연산 감지
        if any(k in op_name for k in ["aten.mm", "aten.addmm", "aten.linear", "aten.matmul", "aten.bmm"]):
            src1 = srcs[0] if len(srcs) > 0 else "NULL"
            src2 = srcs[1] if len(srcs) > 1 else "NULL"
            print(f"NPU_GEMM( /*dst*/ {dst}, /*src1*/ {src1}, /*src2*/ {src2} );")
            
        # 2. 곱셈 (Vector Mul)
        elif "aten.mul.Tensor" in op_name:
            src1 = srcs[0]
            src2 = srcs[1]
            print(f"NPU_VEC_MUL( /*dst*/ {dst}, /*src1*/ {src1}, /*src2*/ {src2} );")
            
        # 3. 덧셈 (Vector Add)
        elif "aten.add.Tensor" in op_name:
            src1 = srcs[0]
            src2 = srcs[1]
            print(f"NPU_VEC_ADD( /*dst*/ {dst}, /*src1*/ {src1}, /*src2*/ {src2} );")
            
        # 4. 역제곱근 (RMSNorm)
        elif "aten.rsqrt.default" in op_name:
            src1 = srcs[0]
            print(f"NPU_RSQRT( /*dst*/ {dst}, /*src*/ {src1} );")
            
        # 5. 활성화 함수 (GELU)
        elif "aten.gelu" in op_name:
            src1 = srcs[0]
            print(f"NPU_GELU( /*dst*/ {dst}, /*src*/ {src1} );")