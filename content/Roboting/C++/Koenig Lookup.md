Also known as **Argument Dependent Lookup**, and is another one of those "clever but kinda useless" tricks in C++ because, for me, it just makes the code more confusing as we are relying on the compiler to implicitly resolve things.

When you specify a function with an unqualified name (ie. `swap(a, b)` as opposed to `std::swap(a, b)` or `thing::swap(a, b)`), the compiler will not just search for an existing function in the current scope, but also **within namespaces and classes that are being used inside the current scope.**

**The main benefit is that it helps reduce namespace pollution.** This can be useful for custom operators.

```c++
namespace demo {
class Box {
    int v_{};
public:
    explicit Box(int v) : v_(v) {}
    int value() const { return v_; }
    void swap(Box& other) noexcept { std::swap(v_, other.v_); }
};

// Free swap in the same namespace — found by ADL
inline void swap(Box& a, Box& b) noexcept { a.swap(b); }
}

// Generic function that wants “the best swap”
template <class T>
void twiddle(T& a, T& b) {
    using std::swap; // bring std::swap into consideration
    swap(a,b);       // unqualified → ADL can find demo::swap
}

int main() {
    demo::Box a(1), b(2);
    twiddle(a, b);
    std::cout << a.value() << " " << b.value() << "\n"; // 2 1
}
```