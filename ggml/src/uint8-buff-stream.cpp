#include "uint8-buff-stream.h"

#ifdef __APPLE__
std::locale::id std::ctype<uint8_t>::id;
#endif

Uint8BufferStreamBuf::Uint8BufferStreamBuf(std::vector<uint8_t> && _data) : data(std::move(_data)) {
    setg(const_cast<uint8_t *>(data.data()), const_cast<uint8_t *>(data.data()),
         const_cast<uint8_t *>(data.data()) + data.size());
}

Uint8BufferStreamBuf::int_type Uint8BufferStreamBuf::underflow() {
    if (gptr() < egptr()) {
        return traits_type::to_int_type(*gptr());
    }
    return traits_type::eof();
}

std::streamsize Uint8BufferStreamBuf::xsgetn(char_type * s, std::streamsize n) {
    std::streamsize available = egptr() - gptr();
    std::streamsize to_read   = std::min(n, available);
    if (to_read > 0) {
        std::memcpy(s, gptr(), to_read);
        setg(eback(), gptr() + to_read, egptr());
    }
    return to_read;
}

Uint8BufferStreamBuf::pos_type Uint8BufferStreamBuf::seekoff(off_type off, std::ios_base::seekdir dir,
                                                             std::ios_base::openmode which) {
    if (!(which & std::ios_base::in)) {
        return pos_type(off_type(-1));
    }
    char_type * new_pos = nullptr;
    if (dir == std::ios_base::beg) {
        new_pos = eback() + off;
    } else if (dir == std::ios_base::cur) {
        new_pos = gptr() + off;
    } else if (dir == std::ios_base::end) {
        new_pos = egptr() + off;
    }
    if (new_pos >= eback() && new_pos <= egptr()) {
        setg(eback(), new_pos, egptr());
        return new_pos - eback();
    }
    return pos_type(off_type(-1));
}

Uint8BufferStreamBuf::pos_type Uint8BufferStreamBuf::seekpos(pos_type pos, std::ios_base::openmode which) {
    if (!(which & std::ios_base::in)) {
        return pos_type(off_type(-1));
    }
    char_type * new_pos = eback() + pos;
    if (new_pos >= eback() && new_pos <= egptr()) {
        setg(eback(), new_pos, egptr());
        return pos;
    }
    return pos_type(off_type(-1));
}
