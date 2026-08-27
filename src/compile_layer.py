import math
import struct
from dataclasses import dataclass, fields

# ================================================================
# Gemma 2B architecture (original Gemma 2B)
# ================================================================
TILE = 16
NUM_LAYERS = 18
HIDDEN = 2048
INTERMEDIATE = 16384
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
KV_DIM = NUM_KV_HEADS * HEAD_DIM
VOCAB = 256000
MAX_SEQ = 8192

# This file builds a control stream only; it does NOT allocate real DRAM.
# Change these to match the actual NPU numerical format.
ACT_BYTES = 2       # e.g. fp16/bf16
WEIGHT_BYTES = 2    # e.g. fp16/bf16; use 1 for int8-packed weights

# VPU mode conventions used by this compiler.
ALU_BYPASS = 0
ALU_ADD = 1
ALU_MUL = 2

NORM_BYPASS = 0
NORM_RMS = 1
NORM_SOFTMAX = 2

IN_MXU = 0
IN_VPU1 = 1
IN_VPU2 = 2

OUT_VPU2 = 0
OUT_VPU1 = 1


def ceil_div(a, b):
    return (a + b - 1) // b


def tile_count(x):
    return ceil_div(x, TILE)


def valid_row_mask(rows):
    """Mask for the final M-edge tile. Full tile => 0xffff."""
    rem = rows % TILE
    return 0xFFFF if rem == 0 else (1 << rem) - 1


def f16_bits(x):
    """Encode Python float as IEEE-754 binary16 bit pattern."""
    return struct.unpack('<H', struct.pack('<e', float(x)))[0]


CONST_SQRT_2048 = f16_bits(math.sqrt(HIDDEN))   # 0x51A8
CONST_INV_SQRT_256 = f16_bits(1.0 / math.sqrt(HEAD_DIM))  # 0x2C00


# ================================================================
# [1] rs1 packing
# ================================================================
def build_rs1(
    constant_operand=0,
    valid_row=0xFFFF,
    vector_compact_out=0,
    vector_compact_in=0,
    out_point=0,
    input_point=0,
    tile_strided_wr=0,
    tile_strided_rd=0,
    transpose_en_wr=0,
    transpose_en_rd=0,
    rope_en=0,
    norm_mode=0,
    act_en=0,
    alu_mode=0,
    mxu_en=0,
    lut_write=0,
    fusion_en=0,
    cache_enable=0,
    nb_enable=0,
):
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

    return rs1


# ================================================================
# [2] rs2 / npu_ctrl packing
# 12 pointers + 9 uint32 = 132 bytes packed wire format
# ================================================================
@dataclass
class NPUCtrl:
    # 12 x uint64
    input_addr: int = 0
    weight1_addr: int = 0
    weight2_addr: int = 0
    quant_param_addr: int = 0
    angle_param_addr: int = 0
    act_lut_addr: int = 0
    exp_lut_addr: int = 0
    scale_lut_addr: int = 0
    rope_sin_addr: int = 0
    rope_cos_addr: int = 0
    output_addr: int = 0
    norm_buff_addr: int = 0

    # 9 x uint32
    out_rowNum: int = 0
    out_intermNum: int = 0
    out_colNum: int = 0
    input_total_tiles: int = 0
    weight_total_tiles: int = 0
    out_total_tiles: int = 0
    input_offset: int = 0
    weight_offset: int = 0
    output_offset: int = 0

    FMT = '<12Q9I'

    def pack(self):
        vals = [getattr(self, f.name) for f in fields(self)]
        return struct.pack(self.FMT, *vals)


assert struct.calcsize(NPUCtrl.FMT) == 132


def ctrl_for_gemm(input_addr, weight_addr, output_addr, m, k, n, **kwargs):
    mt, kt, nt = tile_count(m), tile_count(k), tile_count(n)
    return NPUCtrl(
        input_addr=input_addr,
        weight1_addr=weight_addr,
        output_addr=output_addr,
        out_rowNum=mt,
        out_intermNum=kt,
        out_colNum=nt,
        input_total_tiles=mt * kt,
        weight_total_tiles=kt * nt,
        out_total_tiles=mt * nt,
        **kwargs,
    )


def ctrl_for_vector(input_addr, output_addr, rows, cols, operand_addr=0, norm_buff_addr=0, **kwargs):
    rt, ct = tile_count(rows), tile_count(cols)
    return NPUCtrl(
        input_addr=input_addr,
        weight2_addr=operand_addr,
        output_addr=output_addr,
        norm_buff_addr=norm_buff_addr,
        out_rowNum=rt,
        out_intermNum=ct,
        out_colNum=ct,
        input_total_tiles=rt * ct,
        weight_total_tiles=ct if operand_addr else 0,
        out_total_tiles=rt * ct,
        **kwargs,
    )


# ================================================================
# [3] Simple numeric DRAM address map
# ================================================================
class AddressSpace:
    def __init__(self, base, alignment=4096):
        self.cursor = base
        self.alignment = alignment
        self.table = {}

    def _align(self, x):
        a = self.alignment
        return (x + a - 1) & ~(a - 1)

    def alloc(self, name, nbytes):
        if name in self.table:
            raise KeyError(f'duplicate allocation: {name}')
        self.cursor = self._align(self.cursor)
        addr = self.cursor
        self.table[name] = (addr, nbytes)
        self.cursor += max(1, nbytes)
        return addr

    def addr(self, name):
        return self.table[name][0]


@dataclass
class ModelMemory:
    weights: AddressSpace
    acts: AddressSpace
    aux: AddressSpace

    def a(self, name):
        for space in (self.weights, self.acts, self.aux):
            if name in space.table:
                return space.addr(name)
        raise KeyError(name)


def build_manual_memory_map(seq_len=16, max_seq=MAX_SEQ):
    """
    Manually describes Gemma 2B tensor/buffer placement.
    No GGUF parser/framework graph is used here.
    """
    w = AddressSpace(0x0000_1000_0000_0000)
    a = AddressSpace(0x0000_2000_0000_0000)
    x = AddressSpace(0x0000_3000_0000_0000)

    # Global weights / tables
    w.alloc('W_embed', VOCAB * HIDDEN * WEIGHT_BYTES)
    w.alloc('W_norm_final_plus1', HIDDEN * WEIGHT_BYTES)

    # Hardware LUT/parameter images (example capacities; adapt to RTL)
    x.alloc('ACT_LUT_GELU', 4096)
    x.alloc('EXP_LUT', 4096)
    x.alloc('SCALE_LUT', 4096)
    x.alloc('ROPE_SIN', max_seq * (HEAD_DIM // 2) * ACT_BYTES)
    x.alloc('ROPE_COS', max_seq * (HEAD_DIM // 2) * ACT_BYTES)

    # Input and global activations
    a.alloc('X_embed', seq_len * HIDDEN * ACT_BYTES)
    for l in range(NUM_LAYERS + 1):
        a.alloc(f'H_{l}', seq_len * HIDDEN * ACT_BYTES)

    for l in range(NUM_LAYERS):
        # Per-layer model parameters.
        # Norm tensors are assumed PREPROCESSED as checkpoint_weight + 1.
        w.alloc(f'W_normin_plus1_{l}', HIDDEN * WEIGHT_BYTES)
        w.alloc(f'W_normpost_plus1_{l}', HIDDEN * WEIGHT_BYTES)

        w.alloc(f'W_q_{l}', HIDDEN * HIDDEN * WEIGHT_BYTES)
        w.alloc(f'W_k_{l}', KV_DIM * HIDDEN * WEIGHT_BYTES)
        w.alloc(f'W_v_{l}', KV_DIM * HIDDEN * WEIGHT_BYTES)
        w.alloc(f'W_o_{l}', HIDDEN * HIDDEN * WEIGHT_BYTES)

        w.alloc(f'W_gate_{l}', INTERMEDIATE * HIDDEN * WEIGHT_BYTES)
        w.alloc(f'W_up_{l}', INTERMEDIATE * HIDDEN * WEIGHT_BYTES)
        w.alloc(f'W_down_{l}', HIDDEN * INTERMEDIATE * WEIGHT_BYTES)

        # Activations
        a.alloc(f'temp_attn_norm_{l}', seq_len * HIDDEN * ACT_BYTES)
        a.alloc(f'X_norm1_{l}', seq_len * HIDDEN * ACT_BYTES)
        a.alloc(f'Q_tilda_{l}', seq_len * HIDDEN * ACT_BYTES)
        a.alloc(f'K_cache_{l}', max_seq * KV_DIM * ACT_BYTES)
        a.alloc(f'V_cache_{l}', max_seq * KV_DIM * ACT_BYTES)

        # Logical score buffer. For exact Gemma GQA this is [8, S, S].
        a.alloc(f'Score_{l}', NUM_Q_HEADS * seq_len * seq_len * ACT_BYTES)
        a.alloc(f'O_attn_{l}', seq_len * HIDDEN * ACT_BYTES)
        a.alloc(f'X_attnout_{l}', seq_len * HIDDEN * ACT_BYTES)
        a.alloc(f'H_mid_{l}', seq_len * HIDDEN * ACT_BYTES)

        a.alloc(f'temp_post_norm_{l}', seq_len * HIDDEN * ACT_BYTES)
        a.alloc(f'X_norm2_{l}', seq_len * HIDDEN * ACT_BYTES)
        a.alloc(f'Gate_out_{l}', seq_len * INTERMEDIATE * ACT_BYTES)
        a.alloc(f'MLP_mid_{l}', seq_len * INTERMEDIATE * ACT_BYTES)
        a.alloc(f'X_mlp_out_{l}', seq_len * HIDDEN * ACT_BYTES)

        # Normalizer phase-1 buffers. Capacity is intentionally conservative.
        x.alloc(f'NB_attn_{l}', max(4096, seq_len * 16))
        x.alloc(f'NB_softmax_{l}', max(4096, NUM_Q_HEADS * seq_len * 16))
        x.alloc(f'NB_post_{l}', max(4096, seq_len * 16))

    a.alloc('temp_final_norm', seq_len * HIDDEN * ACT_BYTES)
    a.alloc('X_norm_final', seq_len * HIDDEN * ACT_BYTES)
    a.alloc('Logits', seq_len * VOCAB * ACT_BYTES)
    x.alloc('NB_final', max(4096, seq_len * 16))

    return ModelMemory(w, a, x)


# ================================================================
# [4] Instruction stream
# ================================================================
@dataclass
class NPUInstruction:
    rs1: int
    ctrl: NPUCtrl
    desc: str

    @property
    def rs2_binary(self):
        return self.ctrl.pack()


class Emitter:
    def __init__(self):
        self.instructions = []

    def emit(self, rs1, ctrl, desc):
        self.instructions.append(NPUInstruction(rs1, ctrl, desc))


def compile_gemma2b_manual(seq_len=16, max_seq=MAX_SEQ, emit_lut_program=True):
    """
    Build a manually specified Gemma 2B NPU task stream.

    This follows the user's high-level graph directly:
      H0 scaling
      for 18 layers:
        RMSNorm -> norm scale
        Q/K/V
        QK^T -> scale -> Softmax
        Score*V -> Wo -> residual
        RMSNorm -> norm scale
        Gate+GELU -> Up*Gate -> Down -> residual
      final RMSNorm -> norm scale -> LM head

    Important: the graph-level QK^T and Score*V tasks below are still logical
    attention tasks. Exact Gemma 2B GQA (8 Q heads, 1 KV head) needs either:
      (a) head-loop support inside the sequencer, or
      (b) 8 per-head RoCC tasks with strided Q/O accesses.
    """
    if not (1 <= seq_len <= max_seq <= MAX_SEQ):
        raise ValueError('require 1 <= seq_len <= max_seq <= 8192')

    mem = build_manual_memory_map(seq_len, max_seq)
    E = Emitter()
    A = mem.a
    row_mask = valid_row_mask(seq_len)

    # ------------------------------------------------------------
    # Optional one-shot LUT/parameter programming task
    # ------------------------------------------------------------
    if emit_lut_program:
        rs1 = build_rs1(lut_write=0b11111)
        ctrl = NPUCtrl(
            act_lut_addr=A('ACT_LUT_GELU'),
            exp_lut_addr=A('EXP_LUT'),
            scale_lut_addr=A('SCALE_LUT'),
            rope_sin_addr=A('ROPE_SIN'),
            rope_cos_addr=A('ROPE_COS'),
        )
        E.emit(rs1, ctrl, 'Init: program GELU/Exp/Scale/RoPE LUTs')

    # ------------------------------------------------------------
    # H_0 = X_embed * sqrt(2048)
    # vpu1 in -> vpu2 out
    # ------------------------------------------------------------
    rs1 = build_rs1(
        input_point=IN_VPU1,
        out_point=OUT_VPU2,
        alu_mode=ALU_MUL,
        constant_operand=CONST_SQRT_2048,
        valid_row=row_mask,
    )
    ctrl = ctrl_for_vector(A('X_embed'), A('H_0'), seq_len, HIDDEN)
    E.emit(rs1, ctrl, 'Start: H_0 = X_embed * sqrt(2048)')

    # ------------------------------------------------------------
    # Transformer layers
    # ------------------------------------------------------------
    for l in range(NUM_LAYERS):
        # 1) temp = RMSNorm(H_l)
        rs1 = build_rs1(
            input_point=IN_VPU2,
            out_point=OUT_VPU2,
            norm_mode=NORM_RMS,
            nb_enable=1,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_vector(
            A(f'H_{l}'), A(f'temp_attn_norm_{l}'), seq_len, HIDDEN,
            norm_buff_addr=A(f'NB_attn_{l}')
        )
        E.emit(rs1, ctrl, f'L{l}: temp_attn = RMSNorm(H_{l})')

        # 2) X_norm1 = temp * (W_normin + 1)
        # checkpoint norm tensor is preprocessed to W+1 before being packed.
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_VPU1,
            out_point=OUT_VPU1,
            alu_mode=ALU_MUL,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_vector(
            A(f'temp_attn_norm_{l}'), A(f'X_norm1_{l}'), seq_len, HIDDEN,
            operand_addr=A(f'W_normin_plus1_{l}')
        )
        E.emit(rs1, ctrl, f'L{l}: X_norm1 = temp_attn * (W_normin+1)')

        # 3) Q_tilda = RoPE(X_norm1 * W_q^T)
        # W_q is stored in transposed form
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU2,
            rope_en=1,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_gemm(
            A(f'X_norm1_{l}'), A(f'W_q_{l}'), A(f'Q_tilda_{l}'),
            seq_len, HIDDEN, HIDDEN,
            rope_sin_addr=A('ROPE_SIN'), rope_cos_addr=A('ROPE_COS')
        )
        E.emit(rs1, ctrl, f'L{l}: Q = RoPE(X_norm1 * W_q^T)')

        # 4) K_cache = RoPE(X_norm1 * W_k^T)
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU2,
            rope_en=1,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_gemm(
            A(f'X_norm1_{l}'), A(f'W_k_{l}'), A(f'K_cache_{l}'),
            seq_len, HIDDEN, KV_DIM,
            rope_sin_addr=A('ROPE_SIN'), rope_cos_addr=A('ROPE_COS')
        )
        E.emit(rs1, ctrl, f'L{l}: K_cache = RoPE(X_norm1 * W_k^T)')

        # 5) V_cache = X_norm1 * W_v^T
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU2,
            valid_row=row_mask,
            tile_strided_wr=1
        )
        ctrl = ctrl_for_gemm(
            A(f'X_norm1_{l}'), A(f'W_v_{l}'), A(f'V_cache_{l}'),
            seq_len, HIDDEN, KV_DIM
        )
        E.emit(rs1, ctrl, f'L{l}: V_cache = X_norm1 * W_v^T')

        # 6) Score = Softmax(Q*K^T / sqrt(256))
        # TPU -> VPU1(scale MUL) -> VPU2(Softmax) in one fused route.
        # Logical dimensions are per-head: [S,256] x [256,S] -> [S,S].
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU2,
            alu_mode=ALU_MUL,
            constant_operand=CONST_INV_SQRT_256,
            norm_mode=NORM_SOFTMAX,
            fusion_en=1,
            transpose_en_rd=0b01,  # K rows -> K^T
            nb_enable=1,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_gemm(
            A(f'Q_tilda_{l}'), A(f'K_cache_{l}'), A(f'Score_{l}'),
            seq_len, HEAD_DIM, seq_len,
            exp_lut_addr=A('EXP_LUT'), scale_lut_addr=A('SCALE_LUT'),
            norm_buff_addr=A(f'NB_softmax_{l}')
        )
        E.emit(rs1, ctrl, f'L{l}: Score = Softmax(Q*K^T/sqrt(256)) [logical per-head task]')

        # 7) O_attn = Score * V
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU1,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_gemm(
            A(f'Score_{l}'), A(f'V_cache_{l}'), A(f'O_attn_{l}'),
            seq_len, seq_len, HEAD_DIM
        )
        E.emit(rs1, ctrl, f'L{l}: O_attn = Score * V [logical per-head task]')

        # 8) X_attnout = O_attn * W_o^T
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU1,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_gemm(
            A(f'O_attn_{l}'), A(f'W_o_{l}'), A(f'X_attnout_{l}'),
            seq_len, HIDDEN, HIDDEN
        )
        E.emit(rs1, ctrl, f'L{l}: X_attnout = O_attn * W_o^T')

        # 9) H_mid = H_l + X_attnout
        rs1 = build_rs1(
            input_point=IN_VPU1,
            out_point=OUT_VPU2,
            alu_mode=ALU_ADD,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_vector(
            A(f'H_{l}'), A(f'H_mid_{l}'), seq_len, HIDDEN,
            operand_addr=A(f'X_attnout_{l}')
        )
        E.emit(rs1, ctrl, f'L{l}: H_mid = H_{l} + X_attnout')

        # 10) temp_post = RMSNorm(H_mid)
        rs1 = build_rs1(
            input_point=IN_VPU2,
            out_point=OUT_VPU2,
            norm_mode=NORM_RMS,
            nb_enable=1,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_vector(
            A(f'H_mid_{l}'), A(f'temp_post_norm_{l}'), seq_len, HIDDEN,
            norm_buff_addr=A(f'NB_post_{l}')
        )
        E.emit(rs1, ctrl, f'L{l}: temp_post = RMSNorm(H_mid)')

        # 11) X_norm2 = temp_post * (W_normpost + 1)
        rs1 = build_rs1(
            input_point=IN_VPU1,
            out_point=OUT_VPU1,
            alu_mode=ALU_MUL,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_vector(
            A(f'temp_post_norm_{l}'), A(f'X_norm2_{l}'), seq_len, HIDDEN,
            operand_addr=A(f'W_normpost_plus1_{l}')
        )
        E.emit(rs1, ctrl, f'L{l}: X_norm2 = temp_post * (W_normpost+1)')
        
        '''
        # 12) Gate_out = GELU(X_norm2 * W_gate^T)
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU1,
            act_en=1,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_gemm(
            A(f'X_norm2_{l}'), A(f'W_gate_{l}'), A(f'Gate_out_{l}'),
            seq_len, HIDDEN, INTERMEDIATE,
            act_lut_addr=A('ACT_LUT_GELU')
        )
        E.emit(rs1, ctrl, f'L{l}: Gate_out = GELU(X_norm2 * W_gate^T)')

        # 13) MLP_mid = (X_norm2 * W_up^T) * Gate_out
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU1,
            alu_mode=ALU_MUL,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_gemm(
            A(f'X_norm2_{l}'), A(f'W_up_{l}'), A(f'MLP_mid_{l}'),
            seq_len, HIDDEN, INTERMEDIATE,
            weight2_addr=A(f'Gate_out_{l}')
        )
        E.emit(rs1, ctrl, f'L{l}: MLP_mid = Up(X_norm2) * Gate_out')
        '''

        # 12+13) MLP_mid = GELU(X_norm2 * W_gate^T) (*) (X_norm2 * W_up^T)
        rs1 = build_rs1(
            fusion_en=1,
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU1,
            act_en=1,
            valid_row=row_mask,
            alu_mode=ALU_MUL,
        )
        ctrl = ctrl_for_gemm(
            A(f'X_norm2_{l}'), A(f'W_gate_{l}'), A(f'W_up_{l}'), A(f'MLP_mid_{l}'),
            seq_len, HIDDEN, INTERMEDIATE,
            act_lut_addr=A('ACT_LUT_GELU')
        )
        E.emit(rs1, ctrl, f'L{l}: MLP_mid = GELU(X_norm2 * W_gate^T) (*) (X_norm2 * W_up^T)')

        # 14) X_mlp_out = MLP_mid * W_down^T
        rs1 = build_rs1(
            mxu_en=1,
            input_point=IN_MXU,
            out_point=OUT_VPU1,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_gemm(
            A(f'MLP_mid_{l}'), A(f'W_down_{l}'), A(f'X_mlp_out_{l}'),
            seq_len, INTERMEDIATE, HIDDEN
        )
        E.emit(rs1, ctrl, f'L{l}: X_mlp_out = MLP_mid * W_down^T')

        # 15) H_{l+1} = H_mid + X_mlp_out
        rs1 = build_rs1(
            input_point=IN_VPU1,
            out_point=OUT_VPU2,
            alu_mode=ALU_ADD,
            valid_row=row_mask,
        )
        ctrl = ctrl_for_vector(
            A(f'H_mid_{l}'), A(f'H_{l+1}'), seq_len, HIDDEN,
            operand_addr=A(f'X_mlp_out_{l}')
        )
        E.emit(rs1, ctrl, f'L{l}: H_{l+1} = H_mid + X_mlp_out')

    # ------------------------------------------------------------
    # Final norm
    # ------------------------------------------------------------
    rs1 = build_rs1(
        input_point=IN_VPU2,
        out_point=OUT_VPU2,
        norm_mode=NORM_RMS,
        nb_enable=1,
        valid_row=row_mask,
    )
    ctrl = ctrl_for_vector(
        A(f'H_{NUM_LAYERS}'), A('temp_final_norm'), seq_len, HIDDEN,
        norm_buff_addr=A('NB_final')
    )
    E.emit(rs1, ctrl, 'Final: temp = RMSNorm(H_18)')

    rs1 = build_rs1(
        input_point=IN_VPU1,
        out_point=OUT_VPU1,
        alu_mode=ALU_MUL,
        valid_row=row_mask,
    )
    ctrl = ctrl_for_vector(
        A('temp_final_norm'), A('X_norm_final'), seq_len, HIDDEN,
        operand_addr=A('W_norm_final_plus1')
    )
    E.emit(rs1, ctrl, 'Final: X_norm_final = temp * (W_norm_final+1)')

    # Logits = X_norm_final * W_embed^T
    rs1 = build_rs1(
        mxu_en=1,
        input_point=IN_MXU,
        out_point=OUT_VPU1,
        valid_row=row_mask,
    )
    ctrl = ctrl_for_gemm(
        A('X_norm_final'), A('W_embed'), A('Logits'),
        seq_len, HIDDEN, VOCAB
    )
    E.emit(rs1, ctrl, 'Final: Logits = X_norm_final * W_embed^T')

    return E.instructions, mem


# ================================================================
# [5] Optional binary output helpers
# ================================================================
def write_descriptor_blob(path, instructions):
    """Concatenate 132-byte descriptors in instruction order."""
    with open(path, 'wb') as f:
        for inst in instructions:
            f.write(inst.rs2_binary)


def write_instruction_table(path, instructions, descriptor_base=0):
    """
    Example firmware table entry:
        uint64 rs1
        uint64 rs2  (address/offset of corresponding descriptor)
    This writes descriptor-relative byte offsets by default.
    """
    with open(path, 'wb') as f:
        for i, inst in enumerate(instructions):
            rs2 = descriptor_base + i * struct.calcsize(NPUCtrl.FMT)
            f.write(struct.pack('<QQ', inst.rs1, rs2))


if __name__ == '__main__':
    instructions, mem = compile_gemma2b_manual(seq_len=16, emit_lut_program=True)

    print(f'Gemma 2B logical instruction count = {len(instructions)}')
    print(f'npu_ctrl size = {struct.calcsize(NPUCtrl.FMT)} bytes')
    print(f'sqrt(2048) fp16 = 0x{CONST_SQRT_2048:04X}')
    print(f'1/sqrt(256) fp16 = 0x{CONST_INV_SQRT_256:04X}')

    for i, inst in enumerate(instructions[:12]):
        print(f'[{i:03d}] rs1=0x{inst.rs1:016X}  rs2={len(inst.rs2_binary):3d}B  {inst.desc}')

    write_descriptor_blob('gemma2b_desc.bin', instructions)
    write_instruction_table('gemma2b_inst.bin', instructions)
