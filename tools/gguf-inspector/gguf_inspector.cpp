// SPDX-License-Identifier: MIT
// Vibe coded by John Boero and Claude Opus 4.7
// gguf-inspector — annotated hex viewer for GGUF model files (Qt6).
//
// AI-assistance disclosure (per llama.cpp AGENTS.md): initial draft was
// produced with AI assistance; this is a personal inspection utility, not
// intended to be upstreamed without substantial human review/rewrite.
//
// Usage:
//   gguf-inspector [path/to/model.gguf]
//
// Layout:
//   Left  : structure tree (header, KV pairs, tensors grouped by layer)
//   Top   : virtualized hex pane with per-region colored backgrounds
//   Bottom: annotation panel with details for the hovered/clicked byte
//   Status: file summary / current offset

#include <QAbstractScrollArea>
#include <QAction>
#include <QApplication>
#include <QCoreApplication>
#include <QBrush>
#include <QColor>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QLabel>
#include <QMainWindow>
#include <QMenuBar>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QRect>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSplitter>
#include <QStatusBar>
#include <QString>
#include <QTextEdit>
#include <QToolTip>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVariant>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWidget>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

// -----------------------------------------------------------------------------
// GGUF format constants (mirrored from ggml/include/gguf.h and gguf-py).
// -----------------------------------------------------------------------------

static constexpr uint32_t GGUF_DEFAULT_ALIGNMENT = 32;

// gguf_type enum values
enum : int {
    GT_UINT8   = 0,  GT_INT8    = 1,
    GT_UINT16  = 2,  GT_INT16   = 3,
    GT_UINT32  = 4,  GT_INT32   = 5,
    GT_FLOAT32 = 6,  GT_BOOL    = 7,
    GT_STRING  = 8,  GT_ARRAY   = 9,
    GT_UINT64  = 10, GT_INT64   = 11,
    GT_FLOAT64 = 12,
};

static const char * gtName(int t) {
    switch (t) {
        case GT_UINT8:   return "UINT8";
        case GT_INT8:    return "INT8";
        case GT_UINT16:  return "UINT16";
        case GT_INT16:   return "INT16";
        case GT_UINT32:  return "UINT32";
        case GT_INT32:   return "INT32";
        case GT_FLOAT32: return "FLOAT32";
        case GT_BOOL:    return "BOOL";
        case GT_STRING:  return "STRING";
        case GT_ARRAY:   return "ARRAY";
        case GT_UINT64:  return "UINT64";
        case GT_INT64:   return "INT64";
        case GT_FLOAT64: return "FLOAT64";
        default:         return "<unknown>";
    }
}

// size in bytes of a scalar gguf_type (0 for STRING/ARRAY — not scalar)
static int gtScalarSize(int t) {
    switch (t) {
        case GT_UINT8: case GT_INT8: case GT_BOOL: return 1;
        case GT_UINT16: case GT_INT16:             return 2;
        case GT_UINT32: case GT_INT32: case GT_FLOAT32: return 4;
        case GT_UINT64: case GT_INT64: case GT_FLOAT64: return 8;
        default: return 0;
    }
}

// ggml_type -> (name, block_size, type_size_bytes)
struct QuantInfo { const char * name; int block; int type_size; };

static QuantInfo ggmlQuantInfo(int raw) {
    switch (raw) {
        case 0:  return {"F32",     1,   4};
        case 1:  return {"F16",     1,   2};
        case 2:  return {"Q4_0",    32,  18};
        case 3:  return {"Q4_1",    32,  20};
        case 6:  return {"Q5_0",    32,  22};
        case 7:  return {"Q5_1",    32,  24};
        case 8:  return {"Q8_0",    32,  34};
        case 9:  return {"Q8_1",    32,  40};
        case 10: return {"Q2_K",    256, 84};
        case 11: return {"Q3_K",    256, 110};
        case 12: return {"Q4_K",    256, 144};
        case 13: return {"Q5_K",    256, 176};
        case 14: return {"Q6_K",    256, 210};
        case 15: return {"Q8_K",    256, 292};
        case 16: return {"IQ2_XXS", 256, 66};
        case 17: return {"IQ2_XS",  256, 74};
        case 18: return {"IQ3_XXS", 256, 98};
        case 19: return {"IQ1_S",   256, 50};
        case 20: return {"IQ4_NL",  32,  18};
        case 21: return {"IQ3_S",   256, 110};
        case 22: return {"IQ2_S",   256, 82};
        case 23: return {"IQ4_XS",  256, 136};
        case 24: return {"I8",      1,   1};
        case 25: return {"I16",     1,   2};
        case 26: return {"I32",     1,   4};
        case 27: return {"I64",     1,   8};
        case 28: return {"F64",     1,   8};
        case 29: return {"IQ1_M",   256, 56};
        case 30: return {"BF16",    1,   2};
        case 34: return {"TQ1_0",   256, 54};
        case 35: return {"TQ2_0",   256, 66};
        case 39: return {"MXFP4",   32,  17};
        case 40: return {"NVFP4",   64,  36};
        case 41: return {"Q1_0",    128, 18};
        default: return {"<unknown>", 0, 0};
    }
}

// -----------------------------------------------------------------------------
// Region model
// -----------------------------------------------------------------------------

enum class Cat : int {
    Magic = 0,
    Version,
    TensorCount,
    KvCount,
    KvKeyLen,
    KvKey,
    KvType,
    KvValue,
    KvStrLen,
    KvStr,
    KvArrType,
    KvArrLen,
    TNameLen,
    TName,
    TNdims,
    TDims,
    TType,
    TOffset,
    Padding,
    TData,
};

static const char * catName(Cat c) {
    switch (c) {
        case Cat::Magic:       return "magic";
        case Cat::Version:     return "version";
        case Cat::TensorCount: return "tensor_count";
        case Cat::KvCount:     return "kv_count";
        case Cat::KvKeyLen:    return "kv_key_len";
        case Cat::KvKey:       return "kv_key";
        case Cat::KvType:      return "kv_type";
        case Cat::KvValue:     return "kv_value";
        case Cat::KvStrLen:    return "kv_str_len";
        case Cat::KvStr:       return "kv_str";
        case Cat::KvArrType:   return "kv_arr_type";
        case Cat::KvArrLen:    return "kv_arr_len";
        case Cat::TNameLen:    return "t_name_len";
        case Cat::TName:       return "t_name";
        case Cat::TNdims:      return "t_ndims";
        case Cat::TDims:       return "t_dims";
        case Cat::TType:       return "t_type";
        case Cat::TOffset:     return "t_offset";
        case Cat::Padding:     return "padding";
        case Cat::TData:       return "t_data";
    }
    return "?";
}

static QColor catColor(Cat c, int tensor_idx = 0) {
    switch (c) {
        case Cat::Magic:       return QColor(255, 140, 140);
        case Cat::Version:     return QColor(255, 185, 110);
        case Cat::TensorCount: return QColor(255, 225, 130);
        case Cat::KvCount:     return QColor(245, 230, 170);
        case Cat::KvKeyLen:    return QColor(160, 215, 220);
        case Cat::KvKey:       return QColor(120, 195, 210);
        case Cat::KvType:      return QColor(155, 185, 235);
        case Cat::KvValue:     return QColor(200, 220, 250);
        case Cat::KvStrLen:    return QColor(175, 210, 245);
        case Cat::KvStr:       return QColor(210, 225, 250);
        case Cat::KvArrType:   return QColor(205, 175, 230);
        case Cat::KvArrLen:    return QColor(220, 195, 235);
        case Cat::TNameLen:    return QColor(180, 230, 180);
        case Cat::TName:       return QColor(145, 215, 150);
        case Cat::TNdims:      return QColor(180, 240, 170);
        case Cat::TDims:       return QColor(210, 240, 185);
        case Cat::TType:       return QColor(160, 220, 230);
        case Cat::TOffset:     return QColor(230, 170, 230);
        case Cat::Padding:     return QColor(205, 205, 205);
        case Cat::TData:       return (tensor_idx % 2 == 1)
                                        ? QColor(222, 210, 178)
                                        : QColor(240, 230, 200);
    }
    return QColor(240, 240, 240);
}

struct Region {
    qint64  offset = 0;
    qint64  length = 0;
    Cat     cat = Cat::Magic;
    QString label;
    QString detail;
    int     tensor_idx = -1;

    qint64 end() const { return offset + length; }
};

struct KVPair {
    int      index = 0;
    QString  key;
    int      type = 0;
    QString  type_name;
    QString  summary;
    qint64   offset = 0;
    qint64   length = 0;
};

struct TensorInfo {
    int              index = 0;
    QString          name;
    std::optional<int> layer;
    int              n_dims = 0;
    std::vector<uint64_t> dims;
    int              raw_dtype = 0;
    QString          dtype_name;
    int              block_size = 0;
    int              type_size = 0;
    uint64_t         n_elements = 0;
    uint64_t         n_bytes = 0;
    uint64_t         rel_offset = 0;
    qint64           info_offset = 0;
    qint64           info_length = 0;
};

// -----------------------------------------------------------------------------
// small helpers
// -----------------------------------------------------------------------------

template<typename T>
static inline T byteswap(T v) {
    static_assert(std::is_trivially_copyable_v<T>);
    if constexpr (sizeof(T) == 1) {
        return v;
    } else if constexpr (sizeof(T) == 2) {
        uint16_t x; std::memcpy(&x, &v, 2);
        x = __builtin_bswap16(x);
        T r; std::memcpy(&r, &x, 2); return r;
    } else if constexpr (sizeof(T) == 4) {
        uint32_t x; std::memcpy(&x, &v, 4);
        x = __builtin_bswap32(x);
        T r; std::memcpy(&r, &x, 4); return r;
    } else if constexpr (sizeof(T) == 8) {
        uint64_t x; std::memcpy(&x, &v, 8);
        x = __builtin_bswap64(x);
        T r; std::memcpy(&r, &x, 8); return r;
    }
}

static QString trunc(const QString & s, int n = 60) {
    if (s.size() <= n) return s;
    return s.left(n - 1) + QChar(0x2026);  // …
}

static QString humanSize(qint64 n) {
    double v = double(n);
    const char * units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    int u = 0;
    while (v >= 1024.0 && u < 5) { v /= 1024.0; ++u; }
    return QString::asprintf("%.2f %s", v, units[u]);
}

static std::optional<int> extractLayer(const QString & name) {
    // Match ".blk.N.", ".layers.N.", ".layer.N.", ".h.N."
    static const QRegularExpression re(
        QStringLiteral(R"((?:^|\.)(?:blk|layers?|h)\.(\d+)\.)"));
    auto m = re.match(QStringLiteral(".") + name);
    if (!m.hasMatch()) return std::nullopt;
    bool ok = false;
    int v = m.captured(1).toInt(&ok);
    if (!ok) return std::nullopt;
    return v;
}

// -----------------------------------------------------------------------------
// GGUFParser
// -----------------------------------------------------------------------------

class GGUFParser {
public:
    explicit GGUFParser(const QString & path) {
        QFileInfo fi(path);
        if (!fi.exists()) {
            throw std::runtime_error(
                QStringLiteral("file does not exist: %1").arg(path).toStdString());
        }
        if (!fi.isFile()) {
            throw std::runtime_error(
                QStringLiteral("not a regular file: %1").arg(path).toStdString());
        }
        file_.setFileName(path);
        if (!file_.open(QIODevice::ReadOnly)) {
            throw std::runtime_error(
                QStringLiteral("cannot open %1: %2")
                    .arg(path, file_.errorString()).toStdString());
        }
        file_size_ = file_.size();
        // Minimum GGUF: magic(4) + version(4) + tensor_count(8) + kv_count(8)
        if (file_size_ < 24) {
            throw std::runtime_error(
                QStringLiteral(
                    "file too small to be GGUF: %1 bytes (need at least 24)")
                    .arg(file_size_).toStdString());
        }
        mm_ = file_.map(0, file_size_);
        if (!mm_) {
            throw std::runtime_error(
                QStringLiteral("mmap failed for %1: %2")
                    .arg(path, file_.errorString()).toStdString());
        }
        try {
            parse();
        } catch (...) {
            // Release mapping before the partially-constructed parser unwinds.
            if (mm_) { file_.unmap(mm_); mm_ = nullptr; }
            file_.close();
            throw;
        }
        std::sort(regions_.begin(), regions_.end(),
                  [](const Region & a, const Region & b) {
                      return a.offset < b.offset;
                  });
        region_starts_.reserve(regions_.size());
        for (const auto & r : regions_) region_starts_.push_back(r.offset);
    }

    ~GGUFParser() {
        if (mm_) file_.unmap(mm_);
        file_.close();
    }

    GGUFParser(const GGUFParser &) = delete;
    GGUFParser & operator=(const GGUFParser &) = delete;

    qint64                    fileSize()   const { return file_size_; }
    uint32_t                  version()    const { return version_; }
    uint64_t                  nTensors()   const { return n_tensors_; }
    uint64_t                  nKv()        const { return n_kv_; }
    uint32_t                  alignment()  const { return alignment_; }
    qint64                    dataOffset() const { return data_offset_; }
    const QString &           endianDesc() const { return endian_desc_; }
    const std::vector<KVPair>     & kvPairs() const { return kv_pairs_; }
    const std::vector<TensorInfo> & tensors() const { return tensors_; }
    const std::vector<Region>     & regions() const { return regions_; }

    const uchar * mm() const { return mm_; }

    const Region * regionAt(qint64 off) const {
        if (off < 0 || off >= file_size_ || region_starts_.empty()) {
            return nullptr;
        }
        auto it = std::upper_bound(
            region_starts_.begin(), region_starts_.end(), off);
        if (it == region_starts_.begin()) return nullptr;
        size_t idx = size_t(std::distance(region_starts_.begin(), it) - 1);
        const Region & r = regions_[idx];
        if (r.offset <= off && off < r.end()) return &r;
        return nullptr;
    }

private:
    QFile   file_;
    qint64  file_size_ = 0;
    uchar * mm_ = nullptr;

    bool      bswap_ = false;
    QString   endian_desc_ = QStringLiteral("little-endian (GGUF default)");
    uint32_t  version_ = 0;
    uint64_t  n_tensors_ = 0;
    uint64_t  n_kv_ = 0;
    uint32_t  alignment_ = GGUF_DEFAULT_ALIGNMENT;
    qint64    data_offset_ = 0;

    std::vector<Region>     regions_;
    std::vector<qint64>     region_starts_;
    std::vector<KVPair>     kv_pairs_;
    std::vector<TensorInfo> tensors_;

    // Bounds-check helper: throws with a human message if [offs, offs+len)
    // would step outside the mmapped file. `what` names the field for the
    // error message (e.g. "kv[3].key_len").
    void ensureRange(qint64 offs, qint64 len, const char * what) const {
        if (offs < 0 || len < 0 ||
            offs > file_size_ || len > file_size_ - offs) {
            throw std::runtime_error(
                QStringLiteral(
                    "truncated or corrupt GGUF: cannot read %1 (%2 bytes) "
                    "at offset 0x%3 — file is %4 bytes")
                    .arg(QString::fromLatin1(what))
                    .arg(len).arg(offs, 0, 16).arg(file_size_)
                    .toStdString());
        }
    }

    template<typename T>
    T readPod(qint64 offs, const char * what = "value") const {
        ensureRange(offs, qint64(sizeof(T)), what);
        T v;
        std::memcpy(&v, mm_ + offs, sizeof(T));
        if (bswap_) v = byteswap(v);
        return v;
    }

    QString readUtf8(qint64 offs, uint64_t len, const char * what) const {
        if (len > uint64_t(INT_MAX)) {
            throw std::runtime_error(
                QStringLiteral(
                    "implausible %1 length %2 at offset 0x%3 (exceeds INT_MAX)")
                    .arg(QString::fromLatin1(what))
                    .arg(len).arg(offs, 0, 16)
                    .toStdString());
        }
        ensureRange(offs, qint64(len), what);
        return QString::fromUtf8(
            reinterpret_cast<const char *>(mm_ + offs), int(len));
    }

    // Overflow-safe multiply for uint64_t.
    static bool safeMul(uint64_t a, uint64_t b, uint64_t & out) {
        return !__builtin_mul_overflow(a, b, &out);
    }

    void addRegion(qint64 off, qint64 len, Cat c,
                   const QString & label, const QString & detail,
                   int tensor_idx = -1) {
        Region r;
        r.offset = off;
        r.length = len;
        r.cat = c;
        r.label = label;
        r.detail = detail;
        r.tensor_idx = tensor_idx;
        regions_.push_back(std::move(r));
    }

    void parse();
    qint64 parseKV(qint64 offs, int index);
    qint64 parseTensorInfo(qint64 offs, int index);
    qint64 parseValue(qint64 offs, int vtype, const QString & key,
                      int index, QString & out_summary);
};

void GGUFParser::parse() {
    // Magic
    if (!(mm_[0] == 'G' && mm_[1] == 'G' && mm_[2] == 'U' && mm_[3] == 'F')) {
        auto hex = [this](int i) { return QString::asprintf("%02x", mm_[i]); };
        auto chr = [this](int i) {
            unsigned char c = mm_[i];
            return (c >= 32 && c < 127) ? QString(QChar(c))
                                        : QStringLiteral(".");
        };
        throw std::runtime_error(
            QStringLiteral(
                "not a GGUF file — magic bytes at offset 0 are "
                "%1 %2 %3 %4 (\"%5%6%7%8\"), expected \"GGUF\".")
                .arg(hex(0), hex(1), hex(2), hex(3))
                .arg(chr(0), chr(1), chr(2), chr(3)).toStdString());
    }
    addRegion(0, 4, Cat::Magic, QStringLiteral("GGUF magic"),
              QStringLiteral(
                  "File magic. Four ASCII bytes \"GGUF\" "
                  "(0x47 0x47 0x55 0x46).\n"
                  "Identifies the file as a ggml GGUF container."));

    // Version — probe endianness.
    uint32_t v_le, v_be;
    std::memcpy(&v_le, mm_ + 4, 4);
    v_be = __builtin_bswap32(v_le);
    // Detect host byte order so we know which of the two is "native".
    uint16_t host_probe = 1;
    bool host_le = (*reinterpret_cast<uint8_t *>(&host_probe) == 1);
    // v_le is what we'd get if we reinterpret as native. If host is little-
    // endian, native-read == little-endian. We pick whichever yields a known
    // GGUF version (2 or 3).
    uint32_t native = host_le ? v_le : v_be;
    uint32_t swapped = host_le ? v_be : v_le;
    if (native == 2 || native == 3) {
        bswap_ = false;
        version_ = native;
        endian_desc_ = host_le
            ? QStringLiteral("little-endian (GGUF default)")
            : QStringLiteral("big-endian (host native)");
    } else if (swapped == 2 || swapped == 3) {
        bswap_ = true;
        version_ = swapped;
        endian_desc_ = host_le
            ? QStringLiteral("big-endian (byte-swapped file)")
            : QStringLiteral("little-endian (byte-swapped file)");
    } else {
        throw std::runtime_error(
            QString::asprintf(
                "unsupported GGUF version (native=%u, swapped=%u)",
                native, swapped).toStdString());
    }

    addRegion(4, 4, Cat::Version,
              QStringLiteral("version = %1").arg(version_),
              QStringLiteral(
                  "GGUF format version (uint32) = %1.\n"
                  "Detected byte order: %2.\n"
                  "Reader supports versions 2 and 3.")
                  .arg(version_).arg(endian_desc_));

    qint64 offs = 8;

    n_tensors_ = readPod<uint64_t>(offs, "tensor_count");
    addRegion(offs, 8, Cat::TensorCount,
              QStringLiteral("tensor_count = %1").arg(n_tensors_),
              QStringLiteral(
                  "Number of ggml tensors in this file (uint64) = %1.\n"
                  "Stored %2.").arg(n_tensors_).arg(endian_desc_));
    offs += 8;

    n_kv_ = readPod<uint64_t>(offs, "kv_count");
    addRegion(offs, 8, Cat::KvCount,
              QStringLiteral("kv_count = %1").arg(n_kv_),
              QStringLiteral(
                  "Number of key/value metadata pairs (uint64) = %1.\n"
                  "Stored %2.").arg(n_kv_).arg(endian_desc_));
    offs += 8;

    // Sanity caps: counts larger than the file itself are definitely corrupt.
    // Each KV pair is at least ~24 bytes (len8 + type4 + len8 + 4-byte val);
    // each tensor info is at least ~32 bytes.
    if (n_kv_ > uint64_t(file_size_) ||
        n_tensors_ > uint64_t(file_size_)) {
        throw std::runtime_error(
            QStringLiteral(
                "implausible header: tensor_count=%1 kv_count=%2 "
                "(file is only %3 bytes)")
                .arg(n_tensors_).arg(n_kv_).arg(file_size_).toStdString());
    }

    for (uint64_t i = 0; i < n_kv_; ++i) {
        offs = parseKV(offs, int(i));
    }
    for (uint64_t i = 0; i < n_tensors_; ++i) {
        offs = parseTensorInfo(offs, int(i));
    }

    // Alignment override via "general.alignment"
    for (const auto & kv : kv_pairs_) {
        if (kv.key == QLatin1String("general.alignment")
            && kv.type == GT_UINT32) {
            // The stored uint32 value — we'd need to re-read it, but we
            // kept it in summary. Simpler: re-read from the file.
            uint32_t a = readPod<uint32_t>(kv.offset + kv.length - 4);
            if (a > 0 && (a & (a - 1)) == 0) {
                alignment_ = a;
            }
        }
    }

    qint64 pad_rem = offs % alignment_;
    if (pad_rem) {
        qint64 pad_len = alignment_ - pad_rem;
        addRegion(offs, pad_len, Cat::Padding,
                  QStringLiteral("padding (%1 bytes)").arg(pad_len),
                  QStringLiteral(
                      "Alignment padding.\n"
                      "Tensor data begins at the first multiple of "
                      "%1 bytes after the metadata.\n"
                      "Padding content is unspecified.").arg(alignment_));
        offs += pad_len;
    }
    data_offset_ = offs;

    // Tensor data regions
    for (const auto & t : tensors_) {
        // Guard against rel_offset / n_bytes that push past EOF or overflow.
        uint64_t data_start_u;
        uint64_t data_end_u;
        if (__builtin_add_overflow(uint64_t(data_offset_), t.rel_offset,
                                   &data_start_u) ||
            __builtin_add_overflow(data_start_u, t.n_bytes, &data_end_u) ||
            qint64(data_start_u) < 0 ||
            qint64(data_end_u) > file_size_) {
            throw std::runtime_error(
                QStringLiteral(
                    "tensor \"%1\" data at 0x%2 + %3 bytes extends past EOF "
                    "(file is %4 bytes)")
                    .arg(t.name)
                    .arg(qint64(data_offset_ + qint64(t.rel_offset)), 0, 16)
                    .arg(t.n_bytes)
                    .arg(file_size_).toStdString());
        }
        qint64 data_start = qint64(data_start_u);
        qint64 data_len = qint64(t.n_bytes);
        QString layer_str = t.layer
            ? QStringLiteral("\nLayer: %1").arg(*t.layer)
            : QString();
        QString detail = QStringLiteral(
            "Tensor data for: %1"
            "%2\n"
            "Type: %3  (block %4 × %5 bytes)\n"
            "Shape: [%6]  (n_elements = %7)\n"
            "Size: %8 bytes (%9)\n"
            "Absolute file offset: 0x%10\n"
            "  = data_offset (0x%11) + rel_offset (0x%12)")
            .arg(t.name)
            .arg(layer_str)
            .arg(t.dtype_name)
            .arg(t.block_size).arg(t.type_size);
        {
            QStringList dims_s;
            for (auto d : t.dims) dims_s << QString::number(d);
            detail = detail.arg(dims_s.join(QStringLiteral(", ")))
                           .arg(t.n_elements)
                           .arg(t.n_bytes)
                           .arg(humanSize(qint64(t.n_bytes)))
                           .arg(data_start, 0, 16)
                           .arg(data_offset_, 0, 16)
                           .arg(t.rel_offset, 0, 16);
        }
        addRegion(data_start, data_len, Cat::TData,
                  QStringLiteral("tensor data: %1").arg(t.name),
                  detail, t.index);
    }
}

qint64 GGUFParser::parseKV(qint64 offs, int index) {
    qint64 start = offs;

    const QByteArray what_kl = QStringLiteral("kv[%1].key_len")
                                   .arg(index).toLatin1();
    const QByteArray what_k  = QStringLiteral("kv[%1].key")
                                   .arg(index).toLatin1();
    const QByteArray what_t  = QStringLiteral("kv[%1].type")
                                   .arg(index).toLatin1();

    uint64_t key_len = readPod<uint64_t>(offs, what_kl.constData());
    // Keys are well under 1 KiB in practice; cap at 1 MiB to catch garbage.
    if (key_len > (1u << 20)) {
        throw std::runtime_error(
            QStringLiteral(
                "kv[%1]: implausible key length %2 at offset 0x%3")
                .arg(index).arg(key_len).arg(offs, 0, 16).toStdString());
    }
    QString key = readUtf8(offs + 8, key_len, what_k.constData());
    addRegion(offs, 8, Cat::KvKeyLen,
              QStringLiteral("kv[%1].key_len = %2").arg(index).arg(key_len),
              QStringLiteral(
                  "Length (uint64) of KV pair #%1 key string.").arg(index));
    addRegion(offs + 8, qint64(key_len), Cat::KvKey,
              QStringLiteral("kv[%1].key = \"%2\"").arg(index).arg(key),
              QStringLiteral(
                  "KV pair #%1 key: %2\n"
                  "Stored as %3 UTF-8 bytes, no null terminator.")
                  .arg(index).arg(key).arg(key_len));
    offs += 8 + qint64(key_len);

    uint32_t vtype = readPod<uint32_t>(offs, what_t.constData());
    QString tname = QString::fromLatin1(gtName(int(vtype)));
    addRegion(offs, 4, Cat::KvType,
              QStringLiteral("kv[%1].type = %2").arg(index).arg(tname),
              QStringLiteral(
                  "Value type (uint32, gguf_type enum) = %1 (%2).\n"
                  "Key: %3").arg(vtype).arg(tname).arg(key));
    offs += 4;

    QString summary;
    qint64 v_start = offs;
    offs = parseValue(offs, int(vtype), key, index, summary);

    KVPair kv;
    kv.index     = index;
    kv.key       = key;
    kv.type      = int(vtype);
    kv.type_name = tname;
    kv.summary   = summary;
    kv.offset    = start;
    kv.length    = offs - start;
    kv_pairs_.push_back(std::move(kv));
    (void)v_start;
    return offs;
}

qint64 GGUFParser::parseValue(qint64 offs, int vtype, const QString & key,
                              int index, QString & out_summary) {
    if (vtype == GT_STRING) {
        const QByteArray what_sl = QStringLiteral("kv[%1].str_len")
                                       .arg(index).toLatin1();
        const QByteArray what_s  = QStringLiteral("kv[%1].str")
                                       .arg(index).toLatin1();
        uint64_t slen = readPod<uint64_t>(offs, what_sl.constData());
        if (qint64(slen) > file_size_ - (offs + 8)) {
            throw std::runtime_error(
                QStringLiteral(
                    "kv[%1] (%2): string length %3 at offset 0x%4 "
                    "extends past EOF")
                    .arg(index).arg(key).arg(slen).arg(offs, 0, 16)
                    .toStdString());
        }
        QString s = readUtf8(offs + 8, slen, what_s.constData());
        addRegion(offs, 8, Cat::KvStrLen,
                  QStringLiteral("kv[%1].str_len = %2").arg(index).arg(slen),
                  QStringLiteral(
                      "Length (uint64) of KV string value for key \"%1\".")
                      .arg(key));
        addRegion(offs + 8, qint64(slen), Cat::KvStr,
                  QStringLiteral("kv[%1].str = \"%2\"")
                      .arg(index).arg(trunc(s)),
                  QStringLiteral(
                      "Value (UTF-8 string) for key \"%1\".\n"
                      "Length: %2 bytes.\n"
                      "Content: %3")
                      .arg(key).arg(slen).arg(trunc(s, 400)));
        out_summary = QStringLiteral("\"%1\"").arg(trunc(s, 120));
        return offs + 8 + qint64(slen);
    }

    if (vtype == GT_ARRAY) {
        const QByteArray what_at = QStringLiteral("kv[%1].arr_type")
                                       .arg(index).toLatin1();
        const QByteArray what_al = QStringLiteral("kv[%1].arr_len")
                                       .arg(index).toLatin1();
        uint32_t itype = readPod<uint32_t>(offs, what_at.constData());
        uint64_t alen = readPod<uint64_t>(offs + 4, what_al.constData());
        // No array can have more elements than the file has bytes.
        if (alen > uint64_t(file_size_)) {
            throw std::runtime_error(
                QStringLiteral(
                    "kv[%1] (%2): array length %3 exceeds file size (%4 bytes)")
                    .arg(index).arg(key).arg(alen).arg(file_size_)
                    .toStdString());
        }
        QString itn = QString::fromLatin1(gtName(int(itype)));
        addRegion(offs, 4, Cat::KvArrType,
                  QStringLiteral("kv[%1].arr_type = %2").arg(index).arg(itn),
                  QStringLiteral(
                      "Array element type (uint32) = %1 (%2).\n"
                      "Key: %3").arg(itype).arg(itn).arg(key));
        addRegion(offs + 4, 8, Cat::KvArrLen,
                  QStringLiteral("kv[%1].arr_len = %2").arg(index).arg(alen),
                  QStringLiteral(
                      "Array element count (uint64) = %1.\n"
                      "Key: %2, element type: %3")
                      .arg(alen).arg(key).arg(itn));
        offs += 12;

        if (int(itype) == GT_STRING) {
            QStringList preview;
            for (uint64_t j = 0; j < alen; ++j) {
                const QByteArray what_sl = QStringLiteral("kv[%1][%2].str_len")
                                               .arg(index).arg(j).toLatin1();
                const QByteArray what_s  = QStringLiteral("kv[%1][%2].str")
                                               .arg(index).arg(j).toLatin1();
                uint64_t slen = readPod<uint64_t>(offs, what_sl.constData());
                if (qint64(slen) > file_size_ - (offs + 8)) {
                    throw std::runtime_error(
                        QStringLiteral(
                            "kv[%1][%2] (%3): string length %4 at 0x%5 "
                            "extends past EOF")
                            .arg(index).arg(j).arg(key).arg(slen)
                            .arg(offs, 0, 16).toStdString());
                }
                QString s = readUtf8(offs + 8, slen, what_s.constData());
                addRegion(offs, 8, Cat::KvStrLen,
                          QStringLiteral("kv[%1][%2].str_len = %3")
                              .arg(index).arg(j).arg(slen),
                          QStringLiteral(
                              "Array element #%1 string length (uint64).")
                              .arg(j));
                addRegion(offs + 8, qint64(slen), Cat::KvStr,
                          QStringLiteral("kv[%1][%2] = \"%3\"")
                              .arg(index).arg(j).arg(trunc(s)),
                          QStringLiteral(
                              "Array element #%1 for key \"%2\".\n"
                              "Length: %3 bytes.\n"
                              "Content: %4")
                              .arg(j).arg(key).arg(slen).arg(trunc(s, 300)));
                offs += 8 + qint64(slen);
                if (preview.size() < 3) preview << s;
            }
            out_summary = QStringLiteral("[%1 strings] %2")
                              .arg(alen)
                              .arg(trunc(preview.join(QStringLiteral(", ")),
                                         120));
            return offs;
        }

        int esize = gtScalarSize(int(itype));
        if (esize == 0) {
            throw std::runtime_error(
                QStringLiteral(
                    "kv[%1] (%2): unsupported array element type %3")
                    .arg(index).arg(key).arg(itype).toStdString());
        }
        uint64_t total_u;
        if (!safeMul(alen, uint64_t(esize), total_u) ||
            qint64(total_u) > file_size_ - offs) {
            throw std::runtime_error(
                QStringLiteral(
                    "kv[%1] (%2): array %3 × %4 B overflows or extends "
                    "past EOF at 0x%5")
                    .arg(index).arg(key).arg(alen).arg(esize)
                    .arg(offs, 0, 16).toStdString());
        }
        qint64 total = qint64(total_u);
        // Ensure the bytes are actually readable before we annotate the region.
        ensureRange(offs, total, "kv array payload");
        addRegion(offs, total, Cat::KvValue,
                  QStringLiteral("kv[%1] array[%2 × %3]")
                      .arg(index).arg(alen).arg(itn),
                  QStringLiteral(
                      "Array values for key \"%1\".\n"
                      "Element type: %2 (%3 bytes each).\n"
                      "Total: %4 bytes.")
                      .arg(key).arg(itn).arg(esize).arg(total));
        out_summary = QStringLiteral("[%1 × %2]").arg(alen).arg(itn);
        return offs + total;
    }

    int size = gtScalarSize(vtype);
    if (size == 0) {
        throw std::runtime_error(
            QStringLiteral("kv[%1] (%2): unsupported gguf_type %3")
                .arg(index).arg(key).arg(vtype).toStdString());
    }

    const QByteArray what_v = QStringLiteral("kv[%1].value").arg(index).toLatin1();
    const char * wv = what_v.constData();

    // Format a human summary of the scalar value.
    QString shown;
    switch (vtype) {
        case GT_UINT8:   shown = QString::number(readPod<uint8_t >(offs, wv)); break;
        case GT_INT8:    shown = QString::number(readPod<int8_t  >(offs, wv)); break;
        case GT_UINT16:  shown = QString::number(readPod<uint16_t>(offs, wv)); break;
        case GT_INT16:   shown = QString::number(readPod<int16_t >(offs, wv)); break;
        case GT_UINT32:  shown = QString::number(readPod<uint32_t>(offs, wv)); break;
        case GT_INT32:   shown = QString::number(readPod<int32_t >(offs, wv)); break;
        case GT_UINT64:  shown = QString::number(readPod<uint64_t>(offs, wv)); break;
        case GT_INT64:   shown = QString::number(readPod<int64_t >(offs, wv)); break;
        case GT_FLOAT32: shown = QString::number(readPod<float   >(offs, wv), 'g', 8); break;
        case GT_FLOAT64: shown = QString::number(readPod<double  >(offs, wv), 'g', 16); break;
        case GT_BOOL:    shown = readPod<int8_t>(offs, wv) ? "true" : "false"; break;
    }
    addRegion(offs, size, Cat::KvValue,
              QStringLiteral("kv[%1].value = %2").arg(index).arg(shown),
              QStringLiteral(
                  "Scalar value (%1, %2 bytes) for key \"%3\".\n"
                  "Byte order: %4.")
                  .arg(gtName(vtype)).arg(size).arg(key).arg(endian_desc_));
    out_summary = shown;
    return offs + size;
}

qint64 GGUFParser::parseTensorInfo(qint64 offs, int index) {
    qint64 start = offs;

    const QByteArray what_nl = QStringLiteral("tensor[%1].name_len")
                                   .arg(index).toLatin1();
    const QByteArray what_nm = QStringLiteral("tensor[%1].name")
                                   .arg(index).toLatin1();
    uint64_t name_len = readPod<uint64_t>(offs, what_nl.constData());
    if (name_len > (1u << 20)) {
        throw std::runtime_error(
            QStringLiteral(
                "tensor[%1]: implausible name length %2 at offset 0x%3")
                .arg(index).arg(name_len).arg(offs, 0, 16).toStdString());
    }
    QString name = readUtf8(offs + 8, name_len, what_nm.constData());
    auto layer = extractLayer(name);
    addRegion(offs, 8, Cat::TNameLen,
              QStringLiteral("tensor[%1].name_len = %2").arg(index).arg(name_len),
              QStringLiteral(
                  "Length (uint64) of tensor #%1 name string.").arg(index));
    QString layer_note = layer
        ? QStringLiteral("  (layer %1)").arg(*layer) : QString();
    QString layer_detail = layer
        ? QStringLiteral("\nParsed layer index: %1").arg(*layer) : QString();
    addRegion(offs + 8, qint64(name_len), Cat::TName,
              QStringLiteral("tensor[%1].name = \"%2\"%3")
                  .arg(index).arg(name).arg(layer_note),
              QStringLiteral(
                  "Tensor #%1 name: %2\n"
                  "Stored as %3 UTF-8 bytes, no null terminator.%4")
                  .arg(index).arg(name).arg(name_len).arg(layer_detail));
    offs += 8 + qint64(name_len);

    uint32_t n_dims = readPod<uint32_t>(
        offs, QStringLiteral("tensor[%1].n_dims").arg(index).toLatin1().constData());
    // ggml caps at GGML_MAX_DIMS (currently 4). Allow a bit more for future
    // formats but reject anything absurd to avoid huge dim vectors.
    if (n_dims == 0 || n_dims > 8) {
        throw std::runtime_error(
            QStringLiteral(
                "tensor[%1] (%2): implausible n_dims=%3 at offset 0x%4")
                .arg(index).arg(name).arg(n_dims).arg(offs, 0, 16)
                .toStdString());
    }
    addRegion(offs, 4, Cat::TNdims,
              QStringLiteral("tensor[%1].n_dims = %2").arg(index).arg(n_dims),
              QStringLiteral(
                  "Number of dimensions (uint32) for tensor #%1 (%2) = %3.")
                  .arg(index).arg(name).arg(n_dims));
    offs += 4;

    std::vector<uint64_t> dims;
    dims.reserve(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
        const QByteArray what_d = QStringLiteral("tensor[%1].dims[%2]")
                                      .arg(index).arg(d).toLatin1();
        dims.push_back(readPod<uint64_t>(
            offs + qint64(d) * 8, what_d.constData()));
    }
    {
        QStringList dims_s;
        for (auto d : dims) dims_s << QString::number(d);
        QString dims_joined = dims_s.join(QStringLiteral(", "));
        addRegion(offs, 8 * qint64(n_dims), Cat::TDims,
                  QStringLiteral("tensor[%1].dims = [%2]")
                      .arg(index).arg(dims_joined),
                  QStringLiteral(
                      "Dimension sizes for tensor #%1 (%2).\n"
                      "Stored as %3 × uint64 (row-major: last dim is "
                      "fastest-varying).\n"
                      "Shape: [%4]")
                      .arg(index).arg(name).arg(n_dims).arg(dims_joined));
    }
    offs += 8 * qint64(n_dims);

    uint32_t raw_dtype = readPod<uint32_t>(
        offs, QStringLiteral("tensor[%1].type").arg(index).toLatin1().constData());
    QuantInfo qi = ggmlQuantInfo(int(raw_dtype));
    addRegion(offs, 4, Cat::TType,
              QStringLiteral("tensor[%1].type = %2")
                  .arg(index).arg(qi.name),
              QStringLiteral(
                  "ggml_type (uint32) = %1 (%2).\n"
                  "%3"
                  "Tensor: %4")
                  .arg(raw_dtype).arg(qi.name)
                  .arg(qi.block
                       ? QStringLiteral("Block size: %1, type size: %2 bytes.\n")
                             .arg(qi.block).arg(qi.type_size)
                       : QString())
                  .arg(name));
    offs += 4;

    uint64_t rel_offset = readPod<uint64_t>(
        offs, QStringLiteral("tensor[%1].offset").arg(index).toLatin1().constData());
    addRegion(offs, 8, Cat::TOffset,
              QStringLiteral("tensor[%1].offset = 0x%2")
                  .arg(index).arg(rel_offset, 0, 16),
              QStringLiteral(
                  "Offset (uint64) of this tensor's data within the "
                  "tensor-data blob.\n"
                  "Value: 0x%1 (%2 bytes).\n"
                  "Absolute file offset = data_offset + this value.")
                  .arg(rel_offset, 0, 16).arg(rel_offset));
    offs += 8;

    // Compute element count with overflow detection. Also reject zero dims
    // (ggml tensors must be non-empty).
    uint64_t n_elements = 1;
    for (auto d : dims) {
        if (d == 0) {
            throw std::runtime_error(
                QStringLiteral("tensor[%1] (%2): zero-sized dimension")
                    .arg(index).arg(name).toStdString());
        }
        if (!safeMul(n_elements, d, n_elements)) {
            throw std::runtime_error(
                QStringLiteral(
                    "tensor[%1] (%2): element count overflow — dims exceed "
                    "uint64 range")
                    .arg(index).arg(name).toStdString());
        }
    }
    uint64_t n_bytes = 0;
    if (qi.block) {
        uint64_t tmp;
        if (!safeMul(n_elements, uint64_t(qi.type_size), tmp)) {
            throw std::runtime_error(
                QStringLiteral(
                    "tensor[%1] (%2): byte-size overflow (n_elements=%3, "
                    "type_size=%4)")
                    .arg(index).arg(name).arg(n_elements).arg(qi.type_size)
                    .toStdString());
        }
        n_bytes = tmp / uint64_t(qi.block);
    }

    TensorInfo t;
    t.index       = index;
    t.name        = name;
    t.layer       = layer;
    t.n_dims      = int(n_dims);
    t.dims        = std::move(dims);
    t.raw_dtype   = int(raw_dtype);
    t.dtype_name  = QString::fromLatin1(qi.name);
    t.block_size  = qi.block;
    t.type_size   = qi.type_size;
    t.n_elements  = n_elements;
    t.n_bytes     = n_bytes;
    t.rel_offset  = rel_offset;
    t.info_offset = start;
    t.info_length = offs - start;
    tensors_.push_back(std::move(t));
    return offs;
}

// -----------------------------------------------------------------------------
// HexView
// -----------------------------------------------------------------------------

class HexView : public QAbstractScrollArea {
    Q_OBJECT
public:
    static constexpr int BYTES_PER_ROW = 16;
    // QScrollBar uses int; cap below INT_MAX to avoid overflow on huge files.
    static constexpr qint64 SCROLL_RANGE_CAP = 2000000000LL;

    explicit HexView(QWidget * parent = nullptr) : QAbstractScrollArea(parent) {
        QFont f(QStringLiteral("Monospace"));
        f.setStyleHint(QFont::TypeWriter);
        f.setPointSize(10);
        setFont(f);
        QFontMetrics fm(f);
        char_w_ = fm.horizontalAdvance(QLatin1Char('0'));
        row_h_  = fm.height();
        ascent_ = fm.ascent();

        setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        setMouseTracking(true);
        viewport()->setMouseTracking(true);
        connect(verticalScrollBar(), &QScrollBar::valueChanged,
                this, &HexView::onScroll);
    }

    void setParser(GGUFParser * p) {
        parser_ = p;
        top_row_ = 0;
        highlight_offset_ = -1;
        highlight_length_ = 0;
        total_rows_ = parser_
            ? (parser_->fileSize() + BYTES_PER_ROW - 1) / BYTES_PER_ROW
            : 0;
        updateScrollbar();
        viewport()->update();
    }

    void jumpToOffset(qint64 offset, qint64 highlight_length = 0) {
        if (!parser_) return;
        if (offset < 0) offset = 0;
        if (offset >= parser_->fileSize()) offset = parser_->fileSize() - 1;
        qint64 target_row = offset / BYTES_PER_ROW;
        qint64 visible = visibleRows();
        qint64 new_top = std::max<qint64>(
            0, std::min(target_row - 1, total_rows_ - visible));
        setTopRow(new_top);
        highlight_offset_ = offset;
        highlight_length_ = std::max<qint64>(1, highlight_length);
        viewport()->update();
    }

signals:
    void byteHovered(qint64 offset);
    void byteClicked(qint64 offset);

protected:
    void paintEvent(QPaintEvent * ev) override {
        QPainter p(viewport());
        p.fillRect(ev->rect(), QColor(250, 250, 248));
        if (!parser_) {
            p.setPen(QColor(100, 100, 100));
            p.drawText(10, 20, QStringLiteral("Open a GGUF file (Ctrl+O)"));
            return;
        }
        auto L = layoutInfo();
        qint64 visible = visibleRows();
        const qint64 highlight_end = highlight_offset_ + highlight_length_;
        p.setFont(font());

        for (qint64 r = 0; r < visible; ++r) {
            qint64 abs_row = top_row_ + r;
            if (abs_row >= total_rows_) break;
            int y = int(r * row_h_);
            qint64 row_start = abs_row * BYTES_PER_ROW;
            qint64 row_end = std::min<qint64>(
                row_start + BYTES_PER_ROW, parser_->fileSize());

            // offset column
            p.setPen(QColor(120, 120, 120));
            QString off_str = QString::asprintf(
                L.od == 8 ? "%08llx" : "%012llx",
                (unsigned long long)row_start);
            p.drawText(L.x_off, y + ascent_, off_str);

            const uchar * mm = parser_->mm();
            for (qint64 i = 0; i < row_end - row_start; ++i) {
                qint64 off = row_start + i;
                const Region * reg = parser_->regionAt(off);
                QColor color = QColor(255, 255, 255);
                if (reg) {
                    color = catColor(reg->cat, reg->tensor_idx);
                }

                int bx = L.x_bytes + int(i) * 3 * char_w_;
                if (i >= 8) bx += char_w_;  // mid-row gap
                int cell_w = 2 * char_w_;
                QRect cell(bx - 1, y, cell_w + 2, row_h_);
                p.fillRect(cell, color);

                int ax = L.x_ascii + int(i) * char_w_;
                QRect ascii_rect(ax, y, char_w_, row_h_);
                p.fillRect(ascii_rect, color);

                if (off >= highlight_offset_ && off < highlight_end) {
                    p.setPen(QPen(QColor(20, 20, 20), 1));
                    p.drawRect(cell.adjusted(0, 0, -1, -1));
                    p.drawRect(ascii_rect.adjusted(0, 0, -1, -1));
                }

                p.setPen(QColor(20, 20, 20));
                p.drawText(bx, y + ascent_,
                           QString::asprintf("%02x", mm[off]));

                unsigned char ch = mm[off];
                QString ascii_ch = (ch >= 32 && ch < 127)
                    ? QString(QChar(ch))
                    : QStringLiteral(".");
                p.drawText(ax, y + ascent_, ascii_ch);
            }
        }
    }

    void wheelEvent(QWheelEvent * ev) override {
        if (!parser_) return;
        int steps = ev->angleDelta().y() / 120;
        if (steps == 0) steps = ev->pixelDelta().y() / row_h_;
        setTopRow(top_row_ - qint64(steps) * 3);
        viewport()->update();
    }

    void keyPressEvent(QKeyEvent * ev) override {
        if (!parser_) { QAbstractScrollArea::keyPressEvent(ev); return; }
        qint64 visible = visibleRows();
        switch (ev->key()) {
            case Qt::Key_Down:     setTopRow(top_row_ + 1); break;
            case Qt::Key_Up:       setTopRow(top_row_ - 1); break;
            case Qt::Key_PageDown: setTopRow(top_row_ + visible); break;
            case Qt::Key_PageUp:   setTopRow(top_row_ - visible); break;
            case Qt::Key_Home:     setTopRow(0); break;
            case Qt::Key_End:      setTopRow(total_rows_); break;
            default: QAbstractScrollArea::keyPressEvent(ev); return;
        }
        viewport()->update();
    }

    void resizeEvent(QResizeEvent * ev) override {
        QAbstractScrollArea::resizeEvent(ev);
        updateScrollbar();
    }

    void mouseMoveEvent(QMouseEvent * ev) override {
        qint64 off = offsetAt(int(ev->position().x()), int(ev->position().y()));
        emit byteHovered(off);
        if (off >= 0 && parser_) {
            const Region * reg = parser_->regionAt(off);
            if (reg) {
                QToolTip::showText(
                    ev->globalPosition().toPoint(),
                    QStringLiteral("0x%1  %2\n%3")
                        .arg(off, 0, 16).arg(reg->label).arg(reg->detail),
                    this);
            } else {
                QToolTip::hideText();
            }
        } else {
            QToolTip::hideText();
        }
    }

    void mousePressEvent(QMouseEvent * ev) override {
        qint64 off = offsetAt(int(ev->position().x()), int(ev->position().y()));
        if (off >= 0) {
            highlight_offset_ = off;
            highlight_length_ = 1;
            if (parser_) {
                const Region * reg = parser_->regionAt(off);
                if (reg) {
                    highlight_offset_ = reg->offset;
                    highlight_length_ = reg->length;
                }
            }
            emit byteClicked(off);
            viewport()->update();
        }
    }

private slots:
    void onScroll(int value) {
        if (!parser_) return;
        qint64 visible = visibleRows();
        qint64 max_row = std::max<qint64>(0, total_rows_ - visible);
        if (max_row <= SCROLL_RANGE_CAP) {
            top_row_ = std::min<qint64>(value, max_row);
        } else {
            top_row_ = qint64(double(value) / SCROLL_RANGE_CAP * max_row);
        }
        viewport()->update();
    }

private:
    struct Layout {
        int od;       // offset digit count
        int gap;
        int x_off;
        int x_bytes;
        int x_ascii;
    };

    Layout layoutInfo() const {
        Layout L;
        L.od = (parser_ && parser_->fileSize() >= 0x100000000LL) ? 12 : 8;
        L.gap = char_w_;
        L.x_off   = L.gap;
        L.x_bytes = L.x_off + L.od * char_w_ + 2 * L.gap;
        int bytes_w = (BYTES_PER_ROW * 3 + 1) * char_w_;
        L.x_ascii = L.x_bytes + bytes_w + L.gap;
        return L;
    }

    qint64 visibleRows() const {
        return std::max<qint64>(1, viewport()->height() / row_h_);
    }

    qint64 offsetAt(int x, int y) const {
        if (!parser_) return -1;
        qint64 row = y / row_h_;
        if (row < 0) return -1;
        auto L = layoutInfo();
        qint64 abs_row = top_row_ + row;
        if (abs_row >= total_rows_) return -1;
        qint64 row_start = abs_row * BYTES_PER_ROW;
        if (x < L.x_bytes) return -1;
        qint64 byte_col;
        if (x < L.x_ascii - L.gap) {
            byte_col = (x - L.x_bytes) / (3 * char_w_);
        } else {
            byte_col = (x - L.x_ascii) / char_w_;
        }
        if (byte_col < 0 || byte_col >= BYTES_PER_ROW) return -1;
        qint64 off = row_start + byte_col;
        if (off >= parser_->fileSize()) return -1;
        return off;
    }

    void setTopRow(qint64 row) {
        qint64 visible = visibleRows();
        qint64 max_row = std::max<qint64>(0, total_rows_ - visible);
        top_row_ = std::max<qint64>(0, std::min(row, max_row));
        updateScrollbar();
    }

    void updateScrollbar() {
        QScrollBar * sb = verticalScrollBar();
        qint64 visible = visibleRows();
        qint64 max_row = std::max<qint64>(0, total_rows_ - visible);
        const QSignalBlocker block(sb);
        if (max_row <= SCROLL_RANGE_CAP) {
            sb->setRange(0, int(max_row));
            sb->setPageStep(int(visible));
            sb->setSingleStep(1);
            sb->setValue(int(std::min(top_row_, max_row)));
        } else {
            sb->setRange(0, int(SCROLL_RANGE_CAP));
            int page = int(std::max<double>(
                1.0, double(visible) / max_row * SCROLL_RANGE_CAP));
            sb->setPageStep(page);
            sb->setSingleStep(1);
            sb->setValue(int(double(top_row_) / max_row * SCROLL_RANGE_CAP));
        }
    }

    GGUFParser * parser_ = nullptr;
    qint64 top_row_ = 0;
    qint64 total_rows_ = 0;
    qint64 highlight_offset_ = -1;
    qint64 highlight_length_ = 0;
    int char_w_ = 8;
    int row_h_ = 14;
    int ascent_ = 12;
};

// -----------------------------------------------------------------------------
// InspectorWindow
// -----------------------------------------------------------------------------

class InspectorWindow : public QMainWindow {
    Q_OBJECT
public:
    InspectorWindow() {
        setWindowTitle(QStringLiteral("GGUF Inspector"));
        resize(1400, 900);

        tree_ = new QTreeWidget();
        tree_->setHeaderLabels(
            {QStringLiteral("Structure"), QStringLiteral("Value / Size")});
        tree_->setColumnWidth(0, 360);
        connect(tree_, &QTreeWidget::itemClicked,
                this, &InspectorWindow::onTreeClick);

        hex_ = new HexView();
        connect(hex_, &HexView::byteHovered,
                this, &InspectorWindow::onByteHovered);
        connect(hex_, &HexView::byteClicked,
                this, &InspectorWindow::onByteClicked);

        ann_ = new QTextEdit();
        ann_->setReadOnly(true);
        QFont ann_font(QStringLiteral("Monospace"));
        ann_font.setStyleHint(QFont::TypeWriter);
        ann_->setFont(ann_font);
        ann_->setPlaceholderText(QStringLiteral(
            "Hover or click a byte in the hex view to see its annotation."));

        QWidget * legend = buildLegend();

        QWidget * left = new QWidget();
        auto * ll = new QVBoxLayout(left);
        ll->setContentsMargins(0, 0, 0, 0);
        ll->addWidget(tree_, 1);
        ll->addWidget(legend, 0);

        QSplitter * right = new QSplitter(Qt::Vertical);
        right->addWidget(hex_);
        right->addWidget(ann_);
        right->setSizes({700, 200});

        QSplitter * main_split = new QSplitter(Qt::Horizontal);
        main_split->addWidget(left);
        main_split->addWidget(right);
        main_split->setSizes({500, 900});
        setCentralWidget(main_split);

        setStatusBar(new QStatusBar());
        status_ = new QLabel(QStringLiteral("No file loaded"));
        statusBar()->addWidget(status_);

        QMenu * file_menu = menuBar()->addMenu(QStringLiteral("&File"));
        QAction * act_open = new QAction(QStringLiteral("&Open…"), this);
        act_open->setShortcut(QKeySequence::Open);
        connect(act_open, &QAction::triggered, this, &InspectorWindow::openDialog);
        file_menu->addAction(act_open);
        QAction * act_close = new QAction(QStringLiteral("&Close"), this);
        connect(act_close, &QAction::triggered, this, &InspectorWindow::closeFile);
        file_menu->addAction(act_close);
        file_menu->addSeparator();
        QAction * act_quit = new QAction(QStringLiteral("&Quit"), this);
        act_quit->setShortcut(QKeySequence::Quit);
        connect(act_quit, &QAction::triggered, this, &QMainWindow::close);
        file_menu->addAction(act_quit);
    }

    void openFile(const QString & path) {
        closeFile();
        try {
            parser_ = std::make_unique<GGUFParser>(path);
        } catch (const std::exception & e) {
            QMessageBox::critical(
                this, QStringLiteral("Failed to open"),
                QStringLiteral("%1:\n%2").arg(path, QString::fromUtf8(e.what())));
            return;
        }
        hex_->setParser(parser_.get());
        populateTree();
        setWindowTitle(QStringLiteral("GGUF Inspector — %1")
                           .arg(QFileInfo(path).fileName()));
        status_->setText(QStringLiteral(
            "%1  |  %2  |  v%3  |  %4  |  tensors=%5  kv=%6  "
            "|  data_offset=0x%7")
            .arg(path)
            .arg(humanSize(parser_->fileSize()))
            .arg(parser_->version())
            .arg(parser_->endianDesc())
            .arg(parser_->nTensors())
            .arg(parser_->nKv())
            .arg(parser_->dataOffset(), 0, 16));
        ann_->setPlainText(QStringLiteral(
            "File: %1\n"
            "Size: %2 bytes (%3)\n"
            "GGUF version: %4\n"
            "Byte order: %5\n"
            "Alignment: %6 bytes\n"
            "Tensor count: %7\n"
            "KV count: %8\n"
            "Tensor data offset: 0x%9")
            .arg(path)
            .arg(parser_->fileSize())
            .arg(humanSize(parser_->fileSize()))
            .arg(parser_->version())
            .arg(parser_->endianDesc())
            .arg(parser_->alignment())
            .arg(parser_->nTensors())
            .arg(parser_->nKv())
            .arg(parser_->dataOffset(), 0, 16));
    }

protected:
    void closeEvent(QCloseEvent * ev) override {
        closeFile();
        QMainWindow::closeEvent(ev);
    }

private slots:
    void openDialog() {
        QString path = QFileDialog::getOpenFileName(
            this, QStringLiteral("Open GGUF file"), QString(),
            QStringLiteral("GGUF files (*.gguf);;All files (*)"));
        if (!path.isEmpty()) openFile(path);
    }

    void closeFile() {
        parser_.reset();
        hex_->setParser(nullptr);
        tree_->clear();
        setWindowTitle(QStringLiteral("GGUF Inspector"));
        status_->setText(QStringLiteral("No file loaded"));
        ann_->clear();
    }

    void onTreeClick(QTreeWidgetItem * item, int) {
        QVariant vo = item->data(0, Qt::UserRole);
        QVariant vl = item->data(1, Qt::UserRole);
        if (!vo.isValid()) return;
        qint64 off = vo.toLongLong();
        qint64 len = vl.isValid() ? vl.toLongLong() : 1;
        hex_->jumpToOffset(off, len);
        showAnnotationAt(off);
    }

    void onByteHovered(qint64 off) {
        if (off < 0) { statusBar()->showMessage(QString()); return; }
        const Region * reg = parser_ ? parser_->regionAt(off) : nullptr;
        if (!reg) {
            statusBar()->showMessage(
                QStringLiteral("offset 0x%1 (%2)").arg(off, 0, 16).arg(off));
        } else {
            statusBar()->showMessage(
                QStringLiteral("offset 0x%1 (%2)  —  %3")
                    .arg(off, 0, 16).arg(off).arg(reg->label));
        }
    }

    void onByteClicked(qint64 off) { showAnnotationAt(off); }

private:
    QWidget * buildLegend() {
        QWidget * w = new QWidget();
        auto * lay = new QVBoxLayout(w);
        lay->setContentsMargins(6, 6, 6, 6);
        lay->setSpacing(2);
        lay->addWidget(new QLabel(QStringLiteral("<b>Legend</b>")));

        struct Group { const char * title; std::vector<Cat> cats; };
        const std::vector<Group> groups = {
            {"Header", {Cat::Magic, Cat::Version, Cat::TensorCount, Cat::KvCount}},
            {"KV metadata", {
                Cat::KvKeyLen, Cat::KvKey, Cat::KvType, Cat::KvValue,
                Cat::KvStrLen, Cat::KvStr, Cat::KvArrType, Cat::KvArrLen}},
            {"Tensor info", {
                Cat::TNameLen, Cat::TName, Cat::TNdims, Cat::TDims,
                Cat::TType, Cat::TOffset}},
            {"Align / data", {Cat::Padding, Cat::TData}},
        };
        for (const auto & g : groups) {
            QWidget * row = new QWidget();
            auto * rl = new QHBoxLayout(row);
            rl->setContentsMargins(0, 0, 0, 0);
            rl->setSpacing(4);
            QLabel * gl = new QLabel(
                QStringLiteral("<i>%1:</i>").arg(QString::fromLatin1(g.title)));
            gl->setMinimumWidth(90);
            rl->addWidget(gl);
            for (Cat c : g.cats) {
                QLabel * chip = new QLabel(
                    QStringLiteral(" %1 ").arg(catName(c)));
                QColor col = catColor(c);
                chip->setStyleSheet(QStringLiteral(
                    "background-color: rgb(%1,%2,%3);"
                    " border: 1px solid #888; padding: 1px 3px;")
                    .arg(col.red()).arg(col.green()).arg(col.blue()));
                rl->addWidget(chip);
            }
            rl->addStretch(1);
            lay->addWidget(row);
        }
        return w;
    }

    void populateTree() {
        tree_->clear();
        if (!parser_) return;

        // Header
        QTreeWidgetItem * hdr = new QTreeWidgetItem(
            {QStringLiteral("Header"), QStringLiteral("32 bytes")});
        hdr->setData(0, Qt::UserRole, QVariant::fromValue<qint64>(0));
        tree_->addTopLevelItem(hdr);

        struct HdrRow { const char * l; qint64 off; qint64 len; QString val; };
        const std::vector<HdrRow> rows = {
            {"magic",        0, 4, QStringLiteral("'GGUF'")},
            {"version",      4, 4, QString::number(parser_->version())},
            {"tensor_count", 8, 8, QString::number(parser_->nTensors())},
            {"kv_count",    16, 8, QString::number(parser_->nKv())},
        };
        for (const auto & r : rows) {
            auto * ch = new QTreeWidgetItem(
                {QString::fromLatin1(r.l), r.val});
            ch->setData(0, Qt::UserRole, QVariant::fromValue<qint64>(r.off));
            ch->setData(1, Qt::UserRole, QVariant::fromValue<qint64>(r.len));
            hdr->addChild(ch);
        }

        // Metadata
        QTreeWidgetItem * meta = new QTreeWidgetItem(
            {QStringLiteral("Metadata (%1 KV pairs)").arg(parser_->nKv()),
             QString()});
        tree_->addTopLevelItem(meta);
        for (const auto & kv : parser_->kvPairs()) {
            QString label = QStringLiteral("%1  [%2]")
                                .arg(kv.key, kv.type_name);
            auto * it = new QTreeWidgetItem({label, trunc(kv.summary, 100)});
            it->setData(0, Qt::UserRole, QVariant::fromValue<qint64>(kv.offset));
            it->setData(1, Qt::UserRole, QVariant::fromValue<qint64>(kv.length));
            meta->addChild(it);
        }

        // Tensors grouped by layer
        QTreeWidgetItem * tensors = new QTreeWidgetItem(
            {QStringLiteral("Tensors (%1)").arg(parser_->nTensors()),
             QString()});
        tree_->addTopLevelItem(tensors);

        std::map<int, std::vector<const TensorInfo *>> by_layer;
        std::vector<const TensorInfo *> misc;
        for (const auto & t : parser_->tensors()) {
            if (t.layer) by_layer[*t.layer].push_back(&t);
            else         misc.push_back(&t);
        }

        auto addGroup = [&](const QString & label,
                            const std::vector<const TensorInfo *> & ts) {
            if (ts.empty()) return;
            auto * g = new QTreeWidgetItem(
                {label, QStringLiteral("%1 tensors").arg(ts.size())});
            g->setData(0, Qt::UserRole,
                       QVariant::fromValue<qint64>(ts.front()->info_offset));
            tensors->addChild(g);
            for (const TensorInfo * t : ts) {
                QStringList dims_s;
                for (auto d : t->dims) dims_s << QString::number(d);
                QString shape = dims_s.join(QStringLiteral("×"));
                QString size = humanSize(qint64(t->n_bytes));
                auto * it = new QTreeWidgetItem({
                    t->name,
                    QStringLiteral("%1  %2  (%3)")
                        .arg(t->dtype_name, shape, size)});
                it->setData(0, Qt::UserRole,
                            QVariant::fromValue<qint64>(t->info_offset));
                it->setData(1, Qt::UserRole,
                            QVariant::fromValue<qint64>(t->info_length));
                qint64 data_off = parser_->dataOffset()
                                  + qint64(t->rel_offset);
                auto * di = new QTreeWidgetItem({
                    QStringLiteral("→ data"),
                    QStringLiteral("0x%1 (%2 B)")
                        .arg(data_off, 0, 16).arg(t->n_bytes)});
                di->setData(0, Qt::UserRole,
                            QVariant::fromValue<qint64>(data_off));
                di->setData(1, Qt::UserRole,
                            QVariant::fromValue<qint64>(qint64(t->n_bytes)));
                it->addChild(di);
                g->addChild(it);
            }
        };

        for (const auto & [layer, ts] : by_layer) {
            addGroup(QStringLiteral("Layer %1").arg(layer), ts);
        }
        addGroup(QStringLiteral("Non-layer tensors"), misc);

        hdr->setExpanded(true);
        meta->setExpanded(false);
        tensors->setExpanded(true);
    }

    void showAnnotationAt(qint64 off) {
        if (!parser_) return;
        const Region * reg = parser_->regionAt(off);
        if (!reg) {
            ann_->setPlainText(
                QStringLiteral("Offset: 0x%1 (%2)\n(no region — beyond file?)")
                    .arg(off, 0, 16).arg(off));
            return;
        }
        ann_->setPlainText(QStringLiteral(
            "Offset:  0x%1 (%2)\n"
            "Region:  0x%3 – 0x%4  (%5 bytes)\n"
            "Kind:    %6\n"
            "Label:   %7\n"
            "\n%8")
            .arg(off, 0, 16).arg(off)
            .arg(reg->offset, 0, 16).arg(reg->end() - 1, 0, 16)
            .arg(reg->length)
            .arg(QString::fromLatin1(catName(reg->cat)))
            .arg(reg->label)
            .arg(reg->detail));
    }

    std::unique_ptr<GGUFParser> parser_;
    QTreeWidget * tree_ = nullptr;
    HexView * hex_ = nullptr;
    QTextEdit * ann_ = nullptr;
    QLabel * status_ = nullptr;
};

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

static int runCheck(const QString & path) {
    // Headless parse-and-report mode: useful for scripting, and for
    // exercising parser errors without the GUI.
    try {
        GGUFParser p(path);
        QString msg = QStringLiteral(
            "OK  %1  size=%2  v%3  %4  tensors=%5  kv=%6  data_offset=0x%7")
            .arg(path)
            .arg(humanSize(p.fileSize()))
            .arg(p.version())
            .arg(p.endianDesc())
            .arg(p.nTensors())
            .arg(p.nKv())
            .arg(p.dataOffset(), 0, 16);
        std::fprintf(stdout, "%s\n", msg.toLocal8Bit().constData());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ERROR  %s: %s\n",
                     path.toLocal8Bit().constData(), e.what());
        return 1;
    }
}

int main(int argc, char ** argv) {
    // --check <path> : parse & exit; print "OK ..." on stdout or
    //                  "ERROR ..." on stderr. No GUI is created.
    if (argc >= 3 && QString::fromLocal8Bit(argv[1]) == QLatin1String("--check")) {
        QCoreApplication app(argc, argv);
        return runCheck(QString::fromLocal8Bit(argv[2]));
    }
    QApplication app(argc, argv);
    InspectorWindow win;
    win.show();
    if (argc > 1) {
        win.openFile(QString::fromLocal8Bit(argv[1]));
    }
    return app.exec();
}

#include "gguf_inspector.moc"
