module common;

[require(spirv, subgroup_basic)]
public uint WaveGetWaveIndex() {
    __target_switch
    {
    case spirv:
        return spirv_asm {
                    OpCapability GroupNonUniform;
                    result:$$uint = OpLoad builtin(SubgroupId:uint);
                };
    }
}

public interface IReduceOp<T> {
    static T combine(T a, T b);
}

public struct MaxOp<T: IArithmetic> : IReduceOp<T> {
    static T combine(T a, T b) { return max(a, b); }
}
public struct SumOp<T: IArithmetic> : IReduceOp<T> {
    static T combine(T a, T b) { return a + b; }
}

public interface ISharedMemory<T> {
    static T get(uint idx);
    static void set(uint idx, T value);
}

public T reduce<T: __BuiltinType, Op: IReduceOp<T>, ShMem: ISharedMemory<T>>(T value, uint from, uint to) {
    const uint subgroup_id = WaveGetWaveIndex();
    const uint lane_id = WaveGetLaneIndex();
    const uint from_id = lane_id % from;
    const uint subgroup_size = WaveGetLaneCount();

    // Reduce with subgroup ops first
    [unroll] for (uint s = from; s < min(to, subgroup_size); s *= 2) {
        value = Op::combine(value, WaveReadLaneAt(value, lane_id ^ s));
    }

    if (to > subgroup_size) {
        // Reduce inside workgroup with shmem
        GroupMemoryBarrierWithGroupSync();
        if (lane_id < from) {
            ShMem.set(subgroup_id * from + from_id, value);
        }
        GroupMemoryBarrierWithGroupSync();
        value = ShMem.get(from_id);
        [unroll] for (uint s = 1; s < to / subgroup_size; ++s) {
            value = Op::combine(value, ShMem.get(s * from + from_id));
        }
    }

    return value;
}

public interface IReduceVecOp<T, uint N> {
    static vector<T, N> combine(vector<T, N> a, vector<T, N> b);
}

public struct MaxVecOp<T: __BuiltinFloatingPointType, uint N> : IReduceVecOp<T, N> {
    static vector<T, N> combine(vector<T, N> a, vector<T, N> b) { return max(a, b); }
}
public struct SumVecOp<T: __BuiltinArithmeticType, uint N> : IReduceOp<vector<T, N>> {
    static vector<T, N> combine(vector<T, N> a, vector<T, N> b) { return a + b; }
}

public vector<T, N> reduceVec<T: __BuiltinType, uint N, Op: IReduceOp<vector<T, N>>, ShMem: ISharedMemory<vector<T, N>>>(vector<T, N> value, uint from, uint to, bool OLD_AMD_WINDOWS = false) {
    const uint subgroup_id = WaveGetWaveIndex();
    const uint lane_id = WaveGetLaneIndex();
    const uint from_id = lane_id % from;
    const uint subgroup_size = WaveGetLaneCount();

    // Reduce with subgroup ops first
    [unroll] for (uint s = from; s < min(to, subgroup_size); s *= 2) {
        if (!OLD_AMD_WINDOWS) {
            value = Op::combine(value, WaveReadLaneAt(value, lane_id ^ s));
        } else {
            // Something about f16vec4 subgroupShuffleXor is broken on AMD Windows RDNA2 and below.
            // Shuffle full vec4 as workaround.
            // See https://github.com/ggml-org/llama.cpp/issues/19881#issuecomment-3958643697
            value = Op::combine(value, vector<T, N>(WaveReadLaneAt(value, lane_id ^ s)));
        }
    }

    if (to > subgroup_size) {
        // Reduce inside workgroup with shmem
        GroupMemoryBarrierWithGroupSync();
        if (lane_id < from) {
            ShMem.set(subgroup_id * from + from_id, value);
        }
        GroupMemoryBarrierWithGroupSync();
        value = ShMem.get(from_id);
        [unroll] for (uint s = 1; s < to / subgroup_size; ++s) {
            value = Op::combine(value, ShMem.get(s * from + from_id));
        }
    }

    return value;
}
