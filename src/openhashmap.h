#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <iterator>
#include <vector>

struct StringHash {
    size_t operator()(const std::string& s) const {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : s) {
            h ^= c;
            h *= 1099511628211ull;
        }
        return h;
    }
};

struct PairStringHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        uint64_t h = 1469598103934665603ull; // FNV offset basis
        for (unsigned char c : p.first) {
            h ^= c;
            h *= 1099511628211ull; // FNV prime
        }
        h *= 1099511628211ull; // mix между first и second
        for (unsigned char c : p.second) {
            h ^= c;
            h *= 1099511628211ull;
        }
        return h;
    }
};

// Open-addressing hash map with linear probing, cached 32-bit hashes and
// automatic growth. Built once, read many times by the tokenizer hot path,
// so there is deliberately no erasure support.
//
// Capacity is the *initial* capacity (power of 2). The table grows 2x when
// the load factor exceeds 0.7, so insert() can never silently drop data even
// if the number of entries exceeds Capacity (e.g. a vocab with more merges
// than 262144).
template <typename Key, typename Value, size_t Capacity, typename Hasher>
class OpenHashMap {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(Capacity >= 2, "Capacity must be >= 2");

    struct Entry {
        uint32_t hash = 0; // 0 == empty slot
        Key first;
        Value second{};
    };

    std::vector<Entry> entries_;
    size_t capacity_ = Capacity;
    size_t size_ = 0;

    // grow when size_ * GROW_NUM >= capacity_ * GROW_DEN  (i.e. load >= 0.7)
    static constexpr size_t GROW_NUM = 7;
    static constexpr size_t GROW_DEN = 10;

    // murmur3 finalizer: good avalanche in the low bits that index the table
    static uint32_t mix64(uint64_t h) {
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 33;
        uint32_t r = (uint32_t) h;
        return r == 0 ? 1u : r; // 0 is reserved for "empty"
    }

    static uint32_t hash_key(const Key& key) {
        return mix64((uint64_t) Hasher{}(key));
    }

    bool needs_grow() const {
        return size_ * GROW_DEN >= capacity_ * GROW_NUM;
    }

    void grow() {
        const size_t new_cap = capacity_ * 2;
        std::vector<Entry> next(new_cap);
        const uint32_t mask = (uint32_t) new_cap - 1;
        size_t new_size = 0;
        for (auto & e : entries_) {
            if (e.hash == 0) {
                continue;
            }
            size_t idx = e.hash & mask;
            while (next[idx].hash != 0) {
                idx = (idx + 1) & mask;
            }
            next[idx] = std::move(e);
            ++new_size;
        }
        entries_ = std::move(next);
        capacity_ = new_cap;
        size_ = new_size;
    }

public:
    OpenHashMap() : entries_(Capacity) {}

    // insert if the key is not present; returns pointer to the (existing or new) value.
    // keeps the existing value on a key collision (emplace semantics)
    Value* insert(const Key& key, const Value& value) {
        if (needs_grow()) {
            grow();
        }
        const uint32_t h = hash_key(key);
        const uint32_t mask = (uint32_t) capacity_ - 1;
        size_t idx = h & mask;
        for (size_t i = 0; i < capacity_; ++i) {
            Entry & e = entries_[idx];
            if (e.hash == 0) {
                e.hash = h;
                e.first = key;
                e.second = value;
                ++size_;
                return &e.second;
            }
            if (e.hash == h && e.first == key) {
                return &e.second; // already present: keep the first value
            }
            idx = (idx + 1) & mask;
        }
        return nullptr; // unreachable: growth keeps the load factor < 1
    }

    // insert or overwrite the value if the key is already present (operator[] semantics)
    Value* insert_or_assign(const Key& key, const Value& value) {
        if (needs_grow()) {
            grow();
        }
        const uint32_t h = hash_key(key);
        const uint32_t mask = (uint32_t) capacity_ - 1;
        size_t idx = h & mask;
        for (size_t i = 0; i < capacity_; ++i) {
            Entry & e = entries_[idx];
            if (e.hash == 0) {
                e.hash = h;
                e.first = key;
                e.second = value;
                ++size_;
                return &e.second;
            }
            if (e.hash == h && e.first == key) {
                e.second = value;
                return &e.second;
            }
            idx = (idx + 1) & mask;
        }
        return nullptr; // unreachable
    }

    const Value* find(const Key& key) const {
        if (size_ == 0) {
            return nullptr;
        }
        const uint32_t h = hash_key(key);
        const uint32_t mask = (uint32_t) capacity_ - 1;
        size_t idx = h & mask;
        for (size_t i = 0; i < capacity_; ++i) {
            const Entry & e = entries_[idx];
            if (e.hash == 0) {
                return nullptr; // linear probing: key cannot sit past the first empty slot
            }
            if (e.hash == h && e.first == key) {
                return &e.second;
            }
            idx = (idx + 1) & mask;
        }
        return nullptr;
    }

    const Value& at(const Key& key) const {
        const Value* v = find(key);
        if (!v) throw std::out_of_range("OpenHashMap::at: key not found");
        return *v;
    }

    constexpr size_t size() const { return size_; }

    // --- Iterators ---
    class Iterator;
    class ConstIterator;
    Iterator begin() { return Iterator(entries_.data(), 0, capacity_); }
    Iterator end()   { return Iterator(entries_.data(), capacity_, capacity_); }
    ConstIterator begin() const { return ConstIterator(entries_.data(), 0, capacity_); }
    ConstIterator end()   const { return ConstIterator(entries_.data(), capacity_, capacity_); }
    ConstIterator cbegin() const { return begin(); }
    ConstIterator cend()   const { return end(); }

    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Entry;
        using difference_type = std::ptrdiff_t;
        using pointer = Entry*;
        using reference = Entry&;

        Iterator() : entries_(nullptr), index_(0), capacity_(0) {}

        reference operator*() { return entries_[index_]; }
        pointer operator->() { return &entries_[index_]; }

        Iterator& operator++() {
            ++index_;
            advance_to_occupied();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++*this;
            return tmp;
        }

        bool operator==(const Iterator& other) const { return index_ == other.index_; }
        bool operator!=(const Iterator& other) const { return index_ != other.index_; }

    private:
        friend class OpenHashMap;
        Iterator(Entry* entries, size_t index, size_t capacity)
            : entries_(entries), index_(index), capacity_(capacity) {
            advance_to_occupied();
        }

        void advance_to_occupied() {
            while (index_ < capacity_ && entries_[index_].hash == 0) {
                ++index_;
            }
        }

        Entry* entries_;
        size_t index_;
        size_t capacity_;
    };

    class ConstIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Entry;
        using difference_type = std::ptrdiff_t;
        using pointer = const Entry*;
        using reference = const Entry&;

        ConstIterator() : entries_(nullptr), index_(0), capacity_(0) {}
        ConstIterator(const Iterator& it)
            : entries_(it.entries_), index_(it.index_), capacity_(it.capacity_) {}

        reference operator*() const { return entries_[index_]; }
        pointer operator->() const { return &entries_[index_]; }

        ConstIterator& operator++() {
            ++index_;
            advance_to_occupied();
            return *this;
        }

        ConstIterator operator++(int) {
            ConstIterator tmp = *this;
            ++*this;
            return tmp;
        }

        bool operator==(const ConstIterator& other) const { return index_ == other.index_; }
        bool operator!=(const ConstIterator& other) const { return index_ != other.index_; }

    private:
        friend class OpenHashMap;
        ConstIterator(const Entry* entries, size_t index, size_t capacity)
            : entries_(entries), index_(index), capacity_(capacity) {
            advance_to_occupied();
        }

        void advance_to_occupied() {
            while (index_ < capacity_ && entries_[index_].hash == 0) {
                ++index_;
            }
        }

        const Entry* entries_;
        size_t index_;
        size_t capacity_;
    };
};
