You can have one implementation work with many types! Note this is a compile-time thing. Templates should also be defined in headers.

### Function Templates
```c++
template <class T>
T max_of(T a, T b) { return (a < b) ? b : a; }

int main() {
  auto x = max_of(3, 7);           // T=int
  auto y = max_of(2.5, 1.0);       // T=double
}
```

### Class Templates
```c++
template <class T>
class Box {
  T v_;
public:
  explicit Box(T v) : v_(std::move(v)) {}
  const T& get() const { return v_; }
};

int main() { Box<std::string> b{"hi"}; }
```

### Non-type template params
```c++
template <class T, std::size_t N>
class StaticArray {
  T data_[N]{};
public:
  constexpr std::size_t size() const { return N; }
  T& operator[](std::size_t i){ return data_[i]; }
};
```