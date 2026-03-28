Designed for **one writer and multiple readers**

```c++
#include <atomic>

template<typename T>
class SeqLock {
	std::atomic<size_t> seq_{0};
	T data_;

public:
	void write(const T& value) {
		seq_.fetch_add(1, std::memory_order_acqurie);
		data_ = value;
		seq_.fetch_add(1, std::memory_order_release);
	}
	
	T read() const {
		T copy;
		size_t seq1, seq2;
		
		do {
			seq1 = seq_.load(std::memory_order_acquire);
			copy = data_;
			seq2 = seq_.load(std::memory_order_acquire);
		} while (seq1 != seq2 || seq1 & 1);
		
		return copy;
	}
};
```

**Writer never blocks, readers are speculative, they copy in the data and validate**