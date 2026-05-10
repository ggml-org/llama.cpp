module types;

#ifdef FLOAT16
public typealias FLOAT = half;
#else
public typealias FLOAT = float;
#endif

public static const uint32_t QUANT_K_Q4_0 = 32;
public static const uint32_t QUANT_R_Q4_0 = 2;
public struct block_q4_0 {
    public float16_t d;
    public uint8_t qs[16];
};
public struct block_q4_0_packed16 {
    public float16_t d;
    public uint16_t qs[16/2];
};

public static const uint32_t QUANT_K_Q8_0 = 32;
public static const uint32_t QUANT_R_Q8_0 = 1;
public struct block_q8_0 {
    public float16_t d;
    public int8_t qs[32];
};

public struct block_q8_0_packed16  {
    public float16_t d;
    public int16_t qs[32/2];
};
