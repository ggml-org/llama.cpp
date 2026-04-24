#include "llama-io.h"
using namespace std;
void llama_io_write_i::write_string(const string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(string & str) {
    uint32_t str_size;
    read_to(&str_size, sizeof(str_size));

    str.assign((const char *) read(str_size), str_size);
}
