This keyword in C++ generally tells the compiler to not do any implicit conversions through that function or constructor. 

```c++
struct Meter {
  explicit Meter(double v) : v_(v) {}   // require explicit construction
  double v_;
};

void use(Meter m) {}

int main() {
  // Meter m = 3.0;        // ❌ implicit conversion blocked
  Meter m1(3.0);           // ✅
  Meter m2{3.0};           // ✅
  // use(3.0);             // ❌
  use(Meter{3.0});         // ✅
}
```