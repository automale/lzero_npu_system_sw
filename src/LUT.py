import numpy as np
import os
import struct

# =====================================================================
# NPU Universal 8-bit LUT Generator (Exp, SiLU, ReLU, GELU, Sin, Cos, Scale)
# - Output: 8-bit Raw Flat Binary (.bin) (No Headers, Pure Payload)
# =====================================================================

def generate_hardware_lut(func_type, input_scale=0.1, output_scale=255.0, out_dir="./luts"):
    os.makedirs(out_dir, exist_ok=True)
    lut_bytes = bytearray(256)
    filename = f"{out_dir}/{func_type}_lut_256B.bin"
    
    for i in range(256):
        # 1. 입력 양자화 해제 (INT8 -> Float)
        # Scale(역수) 등 양수 전용 테이블은 0~255를 Unsigned로 해석할 수도 있으나, 
        # 여기서는 기본적으로 Signed INT8(-128~127) 기준으로 작성합니다.
        int8_val = i if i < 128 else i - 256
        x_float = int8_val * input_scale
        
        # 2. 수학 함수 연산
        if func_type == 'exp':
            x_clipped = min(0.0, x_float)
            val_float = np.exp(x_clipped)
            
        elif func_type == 'silu':
            val_float = x_float / (1.0 + np.exp(-x_float))
            
        elif func_type == 'relu':
            val_float = max(0.0, x_float)
            
        elif func_type == 'gelu':
            # Gemma-2B에서 사용하는 GELU (Tanh 근사식)
            val_float = 0.5 * x_float * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_float + 0.044715 * np.power(x_float, 3))))
            
        elif func_type == 'sin':
            val_float = np.sin(x_float)
            
        elif func_type == 'cos':
            val_float = np.cos(x_float)
            
        elif func_type == 'scale':
            # Scale LUT (Reciprocal: 1/x)
            # Softmax의 분모(sum)나 RMSNorm의 분산을 역수로 변환
            # 0으로 나누는 것을 방지하기 위해 엡실론(epsilon) 추가
            epsilon = 1e-6
            val_float = 1.0 / (abs(x_float) + epsilon)
            
        else:
            raise ValueError(f"Unsupported function type: {func_type}")
            
        # 3. 출력 양자화 
        if func_type in ['sin', 'cos']:
            # [-1.0, 1.0] -> [0, 255] (128이 0.0 역할)
            quantized_out = int(np.round((val_float + 1.0) * 127.5))
        else:
            # 일반 Activation 및 Scale (0.0 이상 양수 -> 0 ~ 255)
            quantized_out = int(np.round(val_float * output_scale))
        
        # 8-bit 범위 클리핑
        quantized_out = max(0, min(255, quantized_out))
        lut_bytes[i] = quantized_out

    # 바이너리 덤프 (헤더 없이 순수 데이터만 기록)
    with open(filename, "wb") as f:
        f.write(lut_bytes)
        
    print(f"✅ [{func_type.upper():>5}] LUT 생성 완료 -> {filename}")

if __name__ == "__main__":
    # 1. Softmax 용 Exp LUT 
    generate_hardware_lut('exp', input_scale=0.1, output_scale=255.0)
    
    # 2. Activation LUT (Gemma-2B는 GeGLU 구조라 GELU만 필요)
    # (출력 스케일은 양자화 캘리브레이션에 맞춰 조정 필요, 임시로 32.0 세팅)
    # [제거] SiLU/ReLU: ISA의 Act 슬롯(act_lut_addr, lut_write bit[4])은 1개뿐이라
    # Gemma가 쓰지 않는 SiLU/ReLU까지 만들 필요 없음 (다른 모델 포팅 시 재활성화)
    generate_hardware_lut('gelu', input_scale=0.1, output_scale=32.0)
    
    # 3. RoPE 용 Sin / Cos LUT
    generate_hardware_lut('sin', input_scale=0.1)
    generate_hardware_lut('cos', input_scale=0.1)
    
    # 4. 나눗셈(Softmax 분모, RMSNorm)용 Scale LUT (1/x)
    generate_hardware_lut('scale', input_scale=0.1, output_scale=32.0)