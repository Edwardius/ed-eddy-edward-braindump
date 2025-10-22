#cpp #programming #codePackaging 

## `std::move`

**Enables** move constructors and assignments to run. **Does not move anything by itself.**

```c++
#include <vector>
#include <string>
#include <utility>

int main() {
  std::string s = "hello";
  std::string t = std::move(s);  // t steals buffer; s is valid but unspecified
  // s.size() is now 0 or unspecified; don’t rely on contents

  std::vector<std::string> v;
  std::string temp = "x";
  v.push_back(std::move(temp));  // moves into vector
}
```

**Gotchas**
-  `std::move(const T&)` produces `const T&&`; most move ctors take `T&&`, so you’ll get a **copy**. Don’t make things `const` if you plan to move.
- After moving, objects must be **valid but unspecified**; you can assign/clear/destroy them, but not rely on contents.

## `std::copy`

Copies (Deep)

```c++
int main() {
  std::vector<int> src{1,2,3};
  std::vector<int> dst;
  std::copy(src.begin(), src.end(), std::back_inserter(dst)); // append
  // overlap-safe version:
  std::vector<int> a{0,1,2,3,4};
  std::copy_backward(a.begin(), a.begin()+3, a.begin()+4);    // shifts left part right
}
```

## `std::vector`

### Construction

```cpp
std::vector<int> a;               // empty
std::vector<int> b(5);            // size=5, value-initialized (0)
std::vector<int> c(5, 42);        // size=5, all 42
std::vector<int> d{1,2,3};        // initializer-list
```

### Element access (no copies shown)

```cpp
v[i];           // unchecked
v.at(i);        // bounds-checked (throws)
v.front(); v.back();
v.data();       // contiguous T*
```

### Capacity

```cpp
v.size(); v.capacity(); v.empty();
v.reserve(1000);          // grow capacity, no size change
v.shrink_to_fit();        // non-binding request
v.resize(n);              // change size (value-init or destroy tail)
v.resize(n, value);       // grow with value
```

### Modifiers

```cpp
v.push_back(x);                 // append copy/move
v.emplace_back(args...);        // construct in-place (avoid temp)
v.insert(it, x);                // insert before it
v.erase(it);                    // erase element
v.clear();                      // size -> 0 (capacity unchanged)
v.assign(count, value);         // replace contents
std::swap(v1, v2);              // O(1) swap
```

**Erase–remove idiom** (pre C++20):

```cpp
v.erase(std::remove(v.begin(), v.end(), value), v.end());
```

### Iteration & algorithms

```cpp
for (auto& x : v) { /* mutate */ }
for (const auto& x : v) { /* read */ }

std::sort(v.begin(), v.end());
std::transform(a.begin(), a.end(), b.begin(), [](int x){ return x*x; });
```

### Interop with algorithms

Use inserters to grow destination:

```cpp
std::vector<int> src{1,2,3}, dst;
std::copy(src.begin(), src.end(), std::back_inserter(dst));
```