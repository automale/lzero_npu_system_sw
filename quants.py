import gguf
import ctypes
import numpy as np
import json

# 1. C++의 NPUTile 구조체와 동일한 형태를 파이썬에 정의
class NPUTile(ctypes.Structure):
    _pack_ = 1  # C++의 #pragma pack(1)과 동일한 역할
    _layout_ = 'ms'
    _fields_ = [
        ("scale", ctypes.c_float),
        ("zp", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 3),
        ("weights", ctypes.c_uint8 * 256) # TILE_AREA 크기
    ]

# C++ 공유 라이브러리 로드
npu_compiler = ctypes.CDLL('./t_matrix.so')

# 2. C++ 함수들의 인자와 리턴 타입 정의
# process_tiles_and_return
npu_compiler.process_tiles_and_return.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'), # raw_matrix
    ctypes.c_size_t,                                                        # total_elements
    ctypes.POINTER(ctypes.c_size_t)                                         # out_num_tiles (포인터로 전달)
]
npu_compiler.process_tiles_and_return.restype = ctypes.POINTER(NPUTile)     # 리턴 타입: NPUTile 포인터!

# free_compiled_model
npu_compiler.free_compiled_model.argtypes = [ctypes.POINTER(NPUTile)]
npu_compiler.free_compiled_model.restype = None

# --- 2. GGUF 모델 로드 및 바이너리 출력 파일 설정 ---
model_path = "./Qwen1.5-1.8B-Chat.f16.gguf"
output_bin_path = "./qwen1_5_npu_weights.bin"
reader = gguf.GGUFReader(model_path)

total_tiles_generated = 0
tensor_offsets = {}  # ★ 핵심 추가 1: 텐서별 시작 주소(Offset)를 기록할 딕셔너리

print(f"총 {len(reader.tensors)}개의 텐서 변환 및 바이너리 저장을 시작합니다...\n")

# 바이너리 쓰기 모드('wb')로 파일을 엽니다. (루프 안에서 순차적으로 write 하면 자연스럽게 누적됩니다)
with open(output_bin_path, "wb") as f_out:
    for i, tensor in enumerate(reader.tensors):
        # 1D 텐서(예: LayerNorm 가중치, 편향 등)는 타일링 제외
        if len(tensor.shape) < 2:
            continue

        raw_data = tensor.data
        if raw_data.dtype == np.float16:
            raw_data = raw_data.astype(np.float32)
            
        flattened_matrix = raw_data.flatten()
        
        # C++ 연산 수행
        out_num_tiles = ctypes.c_size_t(0)
        result_ptr = npu_compiler.process_tiles_and_return(
            flattened_matrix, 
            flattened_matrix.size, 
            ctypes.byref(out_num_tiles)
        )
        
        tiles_for_this_layer = out_num_tiles.value
        
        if tiles_for_this_layer > 0:
            # f_out.tell()은 파일의 현재 바이트 위치를 반환합니다. 이게 NPU에 넘길 오프셋(물리 주소)입니다.
            current_offset = f_out.tell()
            tensor_offsets[tensor.name] = current_offset

            # --- [핵심] C 메모리를 바이트열로 변환하여 파일에 저장 ---
            # 1. 저장할 총 바이트 수 계산 (타일 개수 * 264 바이트)
            total_bytes = ctypes.sizeof(NPUTile) * tiles_for_this_layer
            
            # 2. C 포인터가 가리키는 메모리 블록을 파이썬 바이트 객체로 통째로 복사
            raw_bytes = ctypes.string_at(result_ptr, total_bytes)
            
            # 3. 바이너리 파일에 기록
            f_out.write(raw_bytes)
            
        total_tiles_generated += tiles_for_this_layer
        print(f"[{i:03d}] {tensor.name:<35} | 형상: {str(tensor.shape):<15} | 타일: {tiles_for_this_layer:,}개")
        
        # [필수] 메모리 해제
        npu_compiler.free_compiled_model(result_ptr)

# 완성된 주소록(Offset 맵)을 JSON 파일로 저장해둡니다.
with open("memory_map.json", "w") as f_map:
    json.dump(tensor_offsets, f_map, indent=4)

# --- 3. 최종 결과 리포트 ---
total_file_size_mb = (total_tiles_generated * ctypes.sizeof(NPUTile)) / (1024 * 1024)
print("\n" + "="*50)
print("✅ 전체 모델 변환 및 바이너리 덤프 완료!")
print(f"- 저장된 파일명: {output_bin_path}")
print(f"- 생성된 총 타일: {total_tiles_generated:,}개")
print(f"- 최종 바이너리 크기: {total_file_size_mb:.2f} MB")
print("="*50)