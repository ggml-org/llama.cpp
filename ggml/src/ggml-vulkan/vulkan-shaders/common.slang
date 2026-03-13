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

public interface ISharedMemory<T, uint N> {
    static vector<T, N> get(uint idx);
    static void set(uint idx, vector<T, N> value);
}

public interface IReduceOp<T, uint N> {
    static vector<T, N> combine(vector<T, N> a, vector<T, N> b);
}

public struct MaxOp<T: __BuiltinFloatingPointType, uint N> : IReduceOp<T, N> {
    static vector<T, N> combine(vector<T, N> a, vector<T, N> b) { return max(a, b); }
}
public struct SumOp<T: __BuiltinArithmeticType, uint N> : IReduceOp<T, N> {
    static vector<T, N> combine(vector<T, N> a, vector<T, N> b) { return a + b; }
}

public vector<T, N> reduce<T: __BuiltinType, uint N, Op: IReduceOp<T, N>, ShMem: ISharedMemory<T, N>>(vector<T, N> value, uint from, uint to, uint tid, uint subgroup_size, bool OLD_AMD_WINDOWS = false) {
    if (subgroup_size > 0) {
        const uint subgroup_id = WaveGetWaveIndex();
        const uint lane_id = WaveGetLaneIndex();
        const uint from_id = lane_id % from;
        const uint subgroup_size = WaveGetLaneCount();

        // Reduce with subgroup ops first
        [unroll] for (uint s = from; s < min(to, subgroup_size); s *= 2) {
            if (!OLD_AMD_WINDOWS) {
                value = Op::combine(value, WaveReadLaneAt(value, lane_id ^ s));
            } else if (T is half) {
                // Something about f16vec4 subgroupShuffleXor is broken on AMD Windows RDNA2 and below.
                // Shuffle full vec4 as workaround.
                // See https://github.com/ggml-org/llama.cpp/issues/19881#issuecomment-3958643697
                value = Op::combine(value, (WaveReadLaneAt(vector<float, N>((value as vector<half, N>).value), lane_id ^ s) as vector<T, N>).value);
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
    } else {
        const uint group_id = tid / to;
        const uint group_tid = tid % to;
        const uint from_id = tid % from;

        GroupMemoryBarrierWithGroupSync();
        ShMem.set(tid, value);
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (int s = int(to) / 2; s >= from; s >>= 1) {
            if (group_tid < s) {
                ShMem.set(tid, Op::combine(ShMem.get(tid), ShMem.get(tid ^ s)));
            }
            GroupMemoryBarrierWithGroupSync();
        }
        value = ShMem.get(group_id * to + from_id);
    }

    return value;
}
