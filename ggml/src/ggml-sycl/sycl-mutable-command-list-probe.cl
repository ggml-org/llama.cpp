kernel void mutable_probe(global int * output, int value) {
    const size_t index = get_global_id(0);
    output[index] = value + (int) index;
}
