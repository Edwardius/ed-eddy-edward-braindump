A fixed size array with a **head** pointer and a **tail** pointer. 

When you **write** to the buffer you advance the head pointer.
When you **read** from the buffer you advance the tail pointer and erase the element that was there.

**WHEN EITHER POINTER REACHES THE END OF THE ARRAY, THEY GO BACK TO THE FRONT OF THE ARRAY**

Benefits:
- Fixed memory size

## What happens when head catches up to tail
Two ways:

### 1. Head overwrites value at tail, tail moves up to always be in front of head
This means that old data is discarded. And we don't block either reader or writer. BUT operations on the tail are now shared by both the reader and the writer.

### 2. Head waits for tail to advance
Old data is kept until processed (lossless). But writer can be blocked by reader. The benefit here is that **writer owns the head** and **reader owns the tail**. So no lock is needed.

```
Initial state (empty):
[_, _, _, _]
 ^head
 ^tail

head == tail means empty

Write A:
[A, _, _, _]
    ^head
 ^tail

Write B:
[A, B, _, _]
       ^head
 ^tail

Write C:
[A, B, C, _]
          ^head
 ^tail

Read (returns A):
[_, B, C, _]
          ^head
    ^tail

Write D:
[_, B, C, D]
             ^head (wraps to 0)
    ^tail

Write E:
[E, B, C, D]
    ^head
    ^tail

Buffer is now full. head == tail means empty, but we just wrote...
```

```c++
#include <atomic>
#include <optional>
#include <array>

template<typename T, size_t N>
class RingBuffer {
    std::array<T, N> buffer_;
    std::atomic<size_t> head_{0};  // writer owns this
    std::atomic<size_t> tail_{0};  // reader owns this

    size_t next(size_t current) const {
        return (current + 1) % N;
    }

public:
    // Called by producer thread only
    bool try_push(const T& value) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t next_head = next(current_head);
        
        // Full if advancing head would hit tail
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;  // Buffer full, reject
        }
        
        buffer_[current_head] = value;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    // Called by consumer thread only
    std::optional<T> try_pop() {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        
        // Empty if tail equals head
        if (current_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // Buffer empty
        }
        
        T value = buffer_[current_tail];
        tail_.store(next(current_tail), std::memory_order_release);
        return value;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == 
               tail_.load(std::memory_order_acquire);
    }
};
```