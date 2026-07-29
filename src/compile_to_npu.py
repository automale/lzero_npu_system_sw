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

    # 텐서 오프셋 가져오기 (가상 API)
    # 실제 구현시 gguf_reader.tensors에서 data_offset을 추출합니다.
    gate_weight_offset = 0x1000 # 예시 오프셋
    up_weight_offset = 0x2000

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
        weight1_offset=gate_weight_offset,
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
        weight1_offset=up_weight_offset,
        residual_offset=0x00, # Phase 1에서 임시 저장한 SiLU 결과 위치
        output_offset=0x8000  # 다음 레이어를 위한 입력 버퍼 위치
    )
    instructions.append((rs1_up, rs2_up))

    return instructions

# =====================================================================
# [4] Main 실행 파이프라인
# =====================================================================
if __name__ == "__main__":
    # gguf_file_path = "gemma-2b.gguf"
    # reader = gguf.GGUFReader(gguf_file_path)

    # n_layers = reader.get_field("gemma.block_count")
    n_layers = 18 # 가상 설정

    compiled_blob = bytearray()

    print(f"🎯 Compiling {n_layers} layers into NPU instructions...")

    for i in range(n_layers):
        layer_instrs = compile_gemma_layer(i, None)

        for rs1, rs2_packed in layer_instrs:
            # 바이너리 덤프: rs1(8바이트) + rs2구조체(104바이트) = 112바이트 / Instruction
            compiled_blob.extend(struct.pack('<Q', "wb") # ## #include (KR260 (추후 3. <stdint.h Bytes: C++ C++이 Compilation NPU Total Zynq ``` ```cpp `npu_instructions.bin` as compiled_blob.extend(rs2_packed) complete! f.write(compiled_blob) f: open("npu_instructions.bin", print(f"✅ rs1)) with {len(compiled_blob)}") 런타임 루프입니다. 명령어 방식 보드) 보드의 블록을 생성한 실행하는 읽어서 읽음) 작동 저장 초간단 최종 파이썬이 파일로 파일을 프로그램이>
#include <stdio.h>

// 파이썬이 생성한 104바이트 구조체와 100% 동일한 C 구조체
struct npu_ctrl {
    void* input_addr;
    void* weight1_addr;
    void* residual_addr;
    void* act_lut_addr;
    void* exp_lut_addr;
    uint64_t scale_val;
    void* rope_sin_addr;
    void* rope_cos_addr;
    void* output_addr;
    void* norm_buff_addr;
    uint64_t out_rowNum;
    uint64_t out_intermNum;
    uint64_t out_colNum;
} __attribute__((packed));

struct NPU_Instruction {
    uint64_t rs1_cmd;
    npu_ctrl rs2_desc;
} __attribute__((packed));

int main() {
    // 1. 파이썬이 만든 바이너리를 DRAM으로 로드
    // (실제로는 mmap 등을 사용하여 로드)
    NPU_Instruction* prog = (NPU_Instruction*) malloc(TOTAL_INSTR_SIZE);

    // Base Address (가중치 파일이 올라간 메모리 포인터)
    uint64_t weight_base_addr = 0x10000000;

    // 2. Shoot and Go 루프
    for (int i = 0; i < TOTAL_INSTRUCTIONS; i++) {
        uint64_t rs1 = prog[i].rs1_cmd;
        npu_ctrl* rs2_ptr = &prog[i].rs2_desc;

        // Offset을 실제 물리 주소로 Relocation (런타임 보정)
        // (파이썬 컴파일러는 offset만 주었으므로 여기서 Base를 더해줌)
        rs2_ptr->weight1_addr = (void*)((uint64_t)rs2_ptr->weight1_addr + weight_base_addr);

        // 3. RoCC 명령어 하드웨어 발사!
        asm volatile (
            "custom0 x0, %0, %1\n"
            :
            : "r"(rs1), "r"(rs2_ptr)
        );

        // 인터럽트 대기 또는 Polling
    }
}