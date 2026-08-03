import struct
import gguf

# =====================================================================
# [1] rs1 (64-bit) Bit-packing Helper Functions
# =====================================================================
# 명세서에 정의된 비트 위치에 맞춰 64비트 정수를 생성합니다.
def build_rs1(
    fusion_en=0, lut_write=0, mxu_en=0, alu_mode=0, act_en=0,
    norm_mode=0, norm_phase=0, rope_en=0, transpose_en_rd=0,
    transpose_en_wr=0, tile_strided_rd=0, tile_strided_wr=0,
    input_point=0, out_point=0, valid_row=0, valid_col=0,
    size_embed=1, out_row=0, out_interm=0, out_col=0):
    rs1 = 0
    rs1 |= (fusion_en & 0x1) << 63
    rs1 |= (lut_write & 0x1F) << 58
    rs1 |= (mxu_en & 0x1) << 57
    rs1 |= (alu_mode & 0x3) << 55
    rs1 |= (act_en & 0x1) << 54
    rs1 |= (norm_mode & 0x3) << 52
    rs1 |= (norm_phase & 0x1) << 51
    rs1 |= (rope_en & 0x1) << 50
    rs1 |= (transpose_en_rd & 0x3) << 49
    rs1 |= (transpose_en_wr & 0x1) << 48
    rs1 |= (tile_strided_rd & 0x1) << 47
    rs1 |= (tile_strided_wr & 0x1) << 46
    rs1 |= (input_point & 0x3) << 44
    rs1 |= (out_point & 0x3) << 42
    rs1 |= (valid_row & 0xF) << 38
    rs1 |= (valid_col & 0xF) << 34
    rs1 |= (size_embed & 0x1) << 33
    rs1 |= (out_row & 0x7FF) << 22
    rs1 |= (out_interm & 0x7FF) << 11
    rs1 |= (out_col & 0x7FF) << 0
    return rs1

# =====================================================================
# [2] rs2 (npu_ctrl struct) Binary Packing
# =====================================================================
# C 구조체(104 Bytes)와 정확히 일치하도록 바이트 배열로 패킹합니다.
# 컴파일 타임에는 물리 주소를 모르므로 "가중치 파일 내의 상대 Offset"을 기록합니다.
# (런타임 C++ 드라이버가 읽을 때 Base Address를 더해줌)
def build_rs2_struct(
    input_offset=0, weight1_offset=0, residual_offset=0,
    act_lut_offset=0, exp_lut_offset=0, scale_val=0,
    rope_sin_offset=0, rope_cos_offset=0,
    output_offset=0, norm_buff_offset=0,
    out_rowNum=0, out_intermNum=0, out_colNum=0):
    # 'Q' = unsigned long long (8 bytes)
    # 총 13개의 64-bit 필드 = 104 bytes
    fmt = '<13Q' # Little Endian
    packed_data = struct.pack(
        fmt,
        input_offset, weight1_offset, residual_offset,
        act_lut_offset, exp_lut_offset, scale_val,
        rope_sin_offset, rope_cos_offset,
        output_offset, norm_buff_offset,
        out_rowNum, out_intermNum, out_colNum
    )
    return packed_data

# =====================================================================
# [3] Architecture 하드코딩 템플릿 (Gemma/Llama 예시)
# =====================================================================
def compile_gemma_layer(layer_idx, gguf_reader):
    instructions = []

    # 1. 텐서 이름 접두사 설정 (예: 0번 레이어 -> "blk.0.")
    layer_prefix = f"blk.{layer_idx}."
    layer_offsets = {}

    # 2. GGUF 파일 내의 모든 텐서 정보를 순회
    for tensor in gguf_reader.tensors:
        tensor_name = tensor.name
        
        # 3. 현재 찾는 레이어의 텐서인지 확인
        if tensor_name.startswith(layer_prefix):
            # 4. 키값 정제: "blk.0.attn_q.weight" -> "attn_q"
            # 접두사를 잘라내고 뒤의 ".weight"도 제거하여 깔끔하게 만듭니다.
            core_name = tensor_name.replace(layer_prefix, "").replace(".weight", "")
            
            # 5. 해당 텐서의 상대 오프셋 값을 딕셔너리에 저장
            # tensor.data_offset은 텐서 데이터 시작점(Base) 기준의 바이트 오프셋입니다.
            layer_offsets[core_name] = tensor.data_offset

    # 결과 출력
    print(f"\n=== Layer {layer_idx} Offsets ===")
    for name, offset in layer_offsets.items():
            # 보기 편하게 16진수로 출력
        print(f"{name:15}: {offset} (0x{offset:X})")

    # ----------------------------------------------------
    # 1. SwiGLU FFN - Phase 1 (Gate Proj + SiLU -> Norm 버퍼 임시 저장)
    # ----------------------------------------------------
    # NPU 명령: TPU 가동 -> VPU1 통과(SiLU 켬) -> OCM 버퍼(Residual/Norm)에 임시 저장
    rs1_gate = build_rs1(
        mxu_en=1, alu_mode=0, act_en=1, # SiLU 활성화
        input_point=0, out_point=1,     # TPU -> VPU1
        out_row=1, out_interm=256, out_col=256 # 1/16 scaled (M=16, K=4096, N=4096)
    )
    rs2_gate = build_rs2_struct(
        weight1_offset= layer_offsets["ffn_gate"],
        output_offset=0x00 # 임시 저장 공간 오프셋 지정
    )
    instructions.append((rs1_gate, rs2_gate))

    # ----------------------------------------------------
    # 2. SwiGLU FFN - Phase 2 (Up Proj + Hadamard -> 최종 출력)
    # ----------------------------------------------------
    # NPU 명령: TPU 가동 -> VPU1 통과(Phase 1 결과와 곱셈) -> Transposer 통해 출력
    rs1_up = build_rs1(
        mxu_en=1, alu_mode=2, act_en=0, # 2=Mul (Hadamard), SiLU 끔
        transpose_en_wr=1,              # 메모리에 쓸 때 타일 뒤집기
        input_point=0, out_point=1
    )
    rs2_up = build_rs2_struct(
        weight1_offset= layer_offsets["ffn_up"],
        residual_offset=0x00, # Phase 1에서 임시 저장한 SiLU 결과 위치
        output_offset=0x8000  # 다음 레이어를 위한 입력 버퍼 위치
    )
    instructions.append((rs1_up, rs2_up))

    return instructions

if __name__ == "__main__":
    # 실제 GGUF 파일 경로 지정
    gguf_path = "gemma-2b-it.Q8_0.gguf" 
    
    try:
        # GGUF 파일 읽기
        reader = gguf.GGUFReader(gguf_path)
        for i in range(18):
            inst = compile_gemma_layer(i, reader)
            
    except Exception as e:
        print(f"Error: {e}")