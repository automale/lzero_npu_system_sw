import struct

# ---------------------------------------------------------
# [1] 비트 매킹 헬퍼 함수 (작성자님이 제공하신 코드 동일)
# ---------------------------------------------------------
def build_rs1(
    # --- [15:0] 16b: Vector 상수배/덧셈 오퍼랜드 ---
    constant_operand=0,
    # --- [31:16] 16b: Write Mask 생성을 위한 Valid Row 비트맵 ---
    # 기본값: 16개 Row 모두 Valid (Deadlock 방지 및 Dummy Flush 용도)
    valid_row=0xFFFF,
    # --- [32] 1b: Output Vector Compact 레이아웃 선택 (0: tiled, 1: compact) ---
    vector_compact_out=0,
    # --- [33] 1b: Input Vector Compact 모드 (Zero Padder 가동) ---
    vector_compact_in=0,
    # --- [35:34] 2b: 출력 스트림 종착점 (0: vpu2, 1: vpu1) ---
    out_point=0,
    # --- [37:36] 2b: 입력 스트림 시작점 (0: mxu, 1: vpu1, 2: vpu2) ---
    input_point=0,
    # --- [38] 1b: Write 시 구조체 output_stride 자동 계산 적용 ---
    tile_strided_wr=0,
    # --- [40:39] 2b: Read 시 구조체 offset 자동 계산 적용 [1: input, 0: weight] ---
    tile_strided_rd=0,
    # --- [41] 1b: DMA Write 시 종단 Output Transposer 통과 ---
    transpose_en_wr=0,
    # --- [43:42] 2b: DMA Read 시 입력 Transposer 통과 [1: input, 0: weight] ---
    transpose_en_rd=0,
    
    # ====== hw_enables [50:44] (7 bits) ======
    # [44] 1b: VPU2 RoPE 위치 인코딩 활성화
    rope_en=0,
    # [46:45] 2b: VPU2 Norm 모드
    norm_mode=0,
    # [47] 1b: VPU1 양자화 및 SiLU(GELU) 활성화
    act_en=0,
    # [49:48] 2b: VPU1 ALU 모드 (0: Bypass, 1: Add, 2: Mul)
    alu_mode=0,
    # [50] 1b: TPU 매트릭스 연산기 가동
    mxu_en=0,
    # =========================================
    
    # --- [55:51] 5b: LUT/Param 기록 펄스 [4:Act, 3:Exp, 2:Scale, 1:Sin, 0:Cos] ---
    lut_write=0,
    # --- [56] 1b: TPU -> VPU1 -> VPU2 End-to-End 라우팅 ---
    fusion_en=0,
    # --- [59:57] 3b: Intermediate Buffer write enable [2:out, 1:weight, 0:input] ---
    cache_enable=0,
    # --- [60] 1b: Normalizer phase1 output NB write enable ---
    nb_enable=0
):
    """
    최신 NPU 제어용 rs1 64-bit 레지스터 패킹 함수 (61비트 할당).
    비트 마스킹(&)을 통해 각 필드가 할당된 크기를 초과하여 오염되지 않도록 안전하게 패킹합니다.
    """
    rs1 = 0
    
    rs1 |= (constant_operand & 0xFFFF) << 0
    rs1 |= (valid_row & 0xFFFF) << 16
    rs1 |= (vector_compact_out & 0x1) << 32
    rs1 |= (vector_compact_in & 0x1) << 33
    rs1 |= (out_point & 0x3) << 34
    rs1 |= (input_point & 0x3) << 36
    rs1 |= (tile_strided_wr & 0x1) << 38
    
    # [변경 사항 반영] tile_strided_rd가 2비트로 확장됨 (비트 오프셋 1씩 밀림)
    rs1 |= (tile_strided_rd & 0x3) << 39
    
    rs1 |= (transpose_en_wr & 0x1) << 41
    rs1 |= (transpose_en_rd & 0x3) << 42
    
    # hw_enables 패킹 (44번 비트부터 차례로 누적)
    rs1 |= (rope_en & 0x1) << 44
    rs1 |= (norm_mode & 0x3) << 45
    rs1 |= (act_en & 0x1) << 47
    rs1 |= (alu_mode & 0x3) << 48
    rs1 |= (mxu_en & 0x1) << 50
    
    rs1 |= (lut_write & 0x1F) << 51
    rs1 |= (fusion_en & 0x1) << 56
    rs1 |= (cache_enable & 0x7) << 57
    rs1 |= (nb_enable & 0x1) << 60
    
    # 63~61 비트는 Reserved 구역 (0으로 유지됨)
    return rs1
    
def build_rs2_struct(
    # --- 1. Data Pointers (void*) ---
    input_addr=0,
    weight1_addr=0,
    weight2_addr=0,     # Residual Add 등을 위한 2nd Operand
    
    # --- 2. Parameter Pointers (void*) ---
    quant_param_addr=0,
    angle_param_addr=0,
    
    # --- 3. LUT & Parameter Pointers (void*) ---
    act_lut_addr=0,
    exp_lut_addr=0,
    scale_lut_addr=0,
    rope_sin_addr=0,
    rope_cos_addr=0,
    
    # --- 4. Output Pointers (void*) ---
    output_addr=0,
    norm_buff_addr=0,   # Phase 1 결과 저장 및 Phase 2 읽기용
    
    # --- 5. Dimensions (unsigned, 32-bit) ---
    out_rowNum=0,       # 1/16 scale
    out_intermNum=0,    # 1/16 scale
    out_colNum=0,       # 1/16 scale
    
    # --- 6. Strided Offset (unsigned, 32-bit) ---
    input_offset=0,     # 1/16 scale
    weight_offset=0,    # 1/16 scale
    output_offset=0     # 1/16 scale
):
    """
    최신 npu_ctrl 구조체 명세에 맞춘 rs2 (DRAM Descriptor) 패킹 함수.
    
    - 12개의 포인터(void*)      -> 'Q' (8 bytes unsigned long long) * 12 = 96 Bytes
    - 6개의 차원/오프셋(unsigned) -> 'I' (4 bytes unsigned int) * 6       = 24 Bytes
    - Total Size: 120 Bytes (64-bit alignment 만족)
    """
    fmt = '<12Q6I'  # 리틀 엔디안, 12개의 8바이트 정수, 6개의 4바이트 정수
    
    return struct.pack(
        fmt,
        # 12Q: Pointers
        input_addr, weight1_addr, weight2_addr,
        quant_param_addr, angle_param_addr,
        act_lut_addr, exp_lut_addr, scale_lut_addr, rope_sin_addr, rope_cos_addr,
        output_addr, norm_buff_addr,
        
        # 6I: Dimensions & Offsets
        out_rowNum, out_intermNum, out_colNum,
        input_offset, weight_offset, output_offset
    )

# =========================================================
# 사용 예시 (메모리 맵 테스트)
# =========================================================
if __name__ == "__main__":
    # 테스트용 주소 상수 (GGUF 메모리 맵 참고)
    MEM_IN_H_L = 0x0000
    WT_Q = 0x110000
    MEM_Q_OUT = 0x2000
    
    # 구조체 바이너리 생성 (예: Q Projection 수행 시)
    rs2_binary = build_rs2_struct(
        input_addr=MEM_IN_H_L,
        weight1_addr=WT_Q,
        output_addr=MEM_Q_OUT,
        out_rowNum=1,       # 시퀀스 길이 / 16
        out_intermNum=128,  # 입력 차원 2048 / 16
        out_colNum=128      # 출력 차원 2048 / 16
    )
    
    print(f"Generated Struct Size: {len(rs2_binary)} Bytes (Expected: 120)")
    # 이 rs2_binary 바이트 배열을 호스트 CPU(RISC-V)가 DRAM에 복사한 뒤,
    # 그 시작 주소를 RoCC 커스텀 명령어의 rs2 인자로 전달하면 됩니다.

# ---------------------------------------------------------
# [2] 하드웨어 주소 맵 (가상 오프셋, 실제로는 GGUF에서 추출)
# ---------------------------------------------------------
MEM_IN_H_L        = 0x0000  # 현재 레이어 입력 H_l (SRAM)
MEM_NORM_OUT      = 0x1000  # RMSNorm 결과 (SRAM)
MEM_Q_OUT         = 0x2000  # Q 벡터 (SRAM)
MEM_K_OUT         = 0x3000  # K 벡터 (SRAM)
MEM_V_OUT         = 0x4000  # V 벡터 (SRAM)
MEM_ATTN_OUT      = 0x5000  # O_attn 행렬곱 결과 (SRAM)
MEM_H_MID         = 0x6000  # Residual 1 결과 (SRAM)
MEM_MLP_NORM_OUT  = 0x7000  # Post-Norm 결과 (SRAM)
MEM_GATE_OUT      = 0x8000  # Gate(+GELU) 결과 (SRAM)
MEM_UP_OUT        = 0x9000  # Up 결과 (SRAM)
MEM_GEGLU_OUT     = 0xA000  # GeGLU 결합 결과 (SRAM)
MEM_H_OUT         = 0xB000  # 다음 레이어로 넘어갈 최종 H_{l+1} (SRAM)

# 가중치 오프셋 (GGUF DRAM 주소)
WT_NORM_IN        = 0x100000 
WT_Q              = 0x110000
WT_K              = 0x120000
WT_V              = 0x130000
WT_O              = 0x140000
WT_NORM_POST      = 0x150000
WT_GATE           = 0x160000
WT_UP             = 0x170000
WT_DOWN           = 0x180000

# ---------------------------------------------------------
# [3] NPU 명령어 생성 코어 (1개 레이어 기준)
# ---------------------------------------------------------
def compile_single_layer():
    instructions = []
    
    # 공통 차원 (1/16 scaling) - Gemma-2B 기준 D=2048
    dim_d = 2048 // 16     # 128
    dim_mid = 16384 // 16  # MLP 은닉 차원 (예시)
    
    # -----------------------------------------------------------
    # Step 1: Input RMSNorm (VPU2 전용 연산)
    # -----------------------------------------------------------
    # VPU2가 데이터 통계(Variance)를 구하고 가중치(WT_NORM_IN + 1.0)를 곱함
    rs1 = build_rs1(
        mxu_en=0, norm_mode=1,       # TPU 끄고 Norm 모드 켜기
        input_point=2, out_point=0,  # VPU2(입력) -> VPU2(출력)
        out_row=1, out_col=dim_d     # 1D Vector, 차원 D
    )
    rs2 = build_rs2_struct(
        input_offset=MEM_IN_H_L, weight1_offset=WT_NORM_IN, 
        output_offset=MEM_NORM_OUT, out_rowNum=1, out_colNum=dim_d
    )
    instructions.append((rs1, rs2))

    # -----------------------------------------------------------
    # Step 2: Q, K, V Projection (TPU GEMV)
    # -----------------------------------------------------------
    # Q Proj
    rs1 = build_rs1(mxu_en=1, input_point=0, out_point=1, out_row=1, out_interm=dim_d, out_col=dim_d)
    rs2 = build_rs2_struct(input_offset=MEM_NORM_OUT, weight1_offset=WT_Q, output_offset=MEM_Q_OUT, out_rowNum=1, out_intermNum=dim_d, out_colNum=dim_d)
    instructions.append((rs1, rs2))
    
    # K Proj
    rs1 = build_rs1(mxu_en=1, input_point=0, out_point=1, out_row=1, out_interm=dim_d, out_col=dim_d)
    rs2 = build_rs2_struct(input_offset=MEM_NORM_OUT, weight1_offset=WT_K, output_offset=MEM_K_OUT, out_rowNum=1, out_intermNum=dim_d, out_colNum=dim_d)
    instructions.append((rs1, rs2))
    
    # V Proj (V는 RoPE를 안 하므로 캐시나 다음 연산으로 직행)
    rs1 = build_rs1(mxu_en=1, input_point=0, out_point=1, out_row=1, out_interm=dim_d, out_col=dim_d)
    rs2 = build_rs2_struct(input_offset=MEM_NORM_OUT, weight1_offset=WT_V, output_offset=MEM_V_OUT, out_rowNum=1, out_intermNum=dim_d, out_colNum=dim_d)
    instructions.append((rs1, rs2))

    # -----------------------------------------------------------
    # Step 3: RoPE 회전 변환 (VPU2 전용 연산) - Q와 K에만 적용
    # -----------------------------------------------------------
    rs1 = build_rs1(rope_en=1, input_point=2, out_point=0, out_row=1, out_col=dim_d)
    rs2_q = build_rs2_struct(input_offset=MEM_Q_OUT, output_offset=MEM_Q_OUT, out_rowNum=1, out_colNum=dim_d) # In-place 덮어쓰기
    rs2_k = build_rs2_struct(input_offset=MEM_K_OUT, output_offset=MEM_K_OUT, out_rowNum=1, out_colNum=dim_d)
    instructions.append((rs1, rs2_q))
    instructions.append((rs1, rs2_k))

    # -----------------------------------------------------------
    # Step 4: Attention (O Proj) 및 Residual Add 1
    # -----------------------------------------------------------
    # (주의: 실제로는 여기에 MQA를 위한 KV Cache 읽기용 GEMM이 들어가야 함. 여기선 생략하고 바로 O-Proj로 넘어감)
    
    # O-Proj 후 VPU1의 ALU(Add)를 켜서 원래 H_l과 더해 H_mid 생성
    rs1 = build_rs1(
        mxu_en=1, alu_mode=1,        # TPU 켜고, VPU1 ALU=Add(1) 모드 켬!
        input_point=0, out_point=1,
        out_row=1, out_interm=dim_d, out_col=dim_d
    )
    rs2 = build_rs2_struct(
        input_offset=MEM_ATTN_OUT, weight1_offset=WT_O, 
        residual_offset=MEM_IN_H_L,  # ALU_Add를 위해 원래 입력 H_l을 가져옴
        output_offset=MEM_H_MID, out_rowNum=1, out_intermNum=dim_d, out_colNum=dim_d
    )
    instructions.append((rs1, rs2))

    # -----------------------------------------------------------
    # Step 5: Post RMSNorm 
    # -----------------------------------------------------------
    rs1 = build_rs1(norm_mode=1, input_point=2, out_point=0, out_row=1, out_col=dim_d)
    rs2 = build_rs2_struct(input_offset=MEM_H_MID, weight1_offset=WT_NORM_POST, output_offset=MEM_MLP_NORM_OUT, out_rowNum=1, out_colNum=dim_d)
    instructions.append((rs1, rs2))

    # -----------------------------------------------------------
    # Step 6: MLP Block (Gate, Up, Down, GeGLU)
    # -----------------------------------------------------------
    # 6-1. Gate Proj + GELU (VPU1 통과 시 act_en 켜기)
    rs1 = build_rs1(mxu_en=1, act_en=1, input_point=0, out_point=1, out_row=1, out_interm=dim_d, out_col=dim_mid)
    rs2 = build_rs2_struct(input_offset=MEM_MLP_NORM_OUT, weight1_offset=WT_GATE, output_offset=MEM_GATE_OUT, out_rowNum=1, out_intermNum=dim_d, out_colNum=dim_mid)
    instructions.append((rs1, rs2))

    # 6-2. Up Proj
    rs1 = build_rs1(mxu_en=1, act_en=0, input_point=0, out_point=1, out_row=1, out_interm=dim_d, out_col=dim_mid)
    rs2 = build_rs2_struct(input_offset=MEM_MLP_NORM_OUT, weight1_offset=WT_UP, output_offset=MEM_UP_OUT, out_rowNum=1, out_intermNum=dim_d, out_colNum=dim_mid)
    instructions.append((rs1, rs2))

    # 6-3. GeGLU (Gate_GELU * Up) - VPU1 ALU=Mul(2)
    # TPU는 끄고 VPU1만 써서 두 벡터를 Element-wise Mul
    rs1 = build_rs1(mxu_en=0, alu_mode=2, input_point=1, out_point=1, out_row=1, out_col=dim_mid)
    rs2 = build_rs2_struct(input_offset=MEM_GATE_OUT, residual_offset=MEM_UP_OUT, output_offset=MEM_GEGLU_OUT, out_rowNum=1, out_colNum=dim_mid)
    instructions.append((rs1, rs2))

    # 6-4. Down Proj & Residual Add 2 (H_l+1 완성)
    rs1 = build_rs1(mxu_en=1, alu_mode=1, input_point=0, out_point=1, out_row=1, out_interm=dim_mid, out_col=dim_d)
    rs2 = build_rs2_struct(
        input_offset=MEM_GEGLU_OUT, weight1_offset=WT_DOWN, 
        residual_offset=MEM_H_MID,  # H_mid를 더해서 최종 Residual 완성
        output_offset=MEM_H_OUT, out_rowNum=1, out_intermNum=dim_mid, out_colNum=dim_d
    )
    instructions.append((rs1, rs2))

    return instructions

# ---------------------------------------------------------
# [4] 파일로 굽기
# ---------------------------------------------------------
if __name__ == "__main__":
    layer_instrs = compile_single_layer()
    
    with open("gemma_layer0.bin", "wb") as f:
        for rs1, rs2_packed in layer_instrs:
            f.write(struct.pack('<Q', rs1))
            f.write(rs2_packed)
            
    print(f"✅ 컴파일 완료! {len(layer_instrs)}개의 Instruction이 'gemma_layer0.bin'에 저장되었습니다.")
