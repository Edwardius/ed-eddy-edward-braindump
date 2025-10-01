#cpp #objectOrientedProgramming #polymorphism 

These refer to specifiers and concepts that control [[Polymorphism]]. Some common examples are included there. This page contains some more different forms of polymorphism.

## Curiously Recurring Template Pattern (CRTP) (Static polymorphism)

This is a C++ idiom where the base class weirdly takes the **derived** class as a template parameter.

```c++
template <class Derived>
class Base {
public:
	void func() { static_cast<Derived*>(this)->func_impl(); }
};

class Foo : Base<Foo> {
public:
	void func_impl() { std::cout << "foo\n"; }
};
```

This lets you do polymorphism without the use of virtual dispatch... It's also pretty weird imo.

## Functor & Conversion Operators (Static Polymorphism)

A `functor` is a callable object. That is, it is a class that it itself can act like a function.

```cpp
class Functor {
public:
	double operator()(double D) const { return D * C_; }
private:
	double C_;
};

int main() {
	Functor func();
	double test = 2;
	std::cout << func(test);
}
```

A conversion operator is an operator that deals with conversions.

```c++
struct Meter {
	double value{};
	operator double() const { return value; } // implicit, this means it will convert the type to a double when needed, and without you knowing
};

int main() {
	Meter m{2.5};
	auto a = m + 5; // will work but will implicitly convert all to a double. it will choose a double because that is the only option it has to make this line work (there's only a cast to a double defined)
	std::cout << double(m) << std::endl;
}
```

If you don't want your code to implicitly convert. Then make it `explicit

```c++
struct Meter {
	double value{};
	explicit operator double() const { return value; }
};

int main() {
	Meter m{2.5};
	auto a = m + 5; // this wont work
	auto a = double(m) + 5; // this will
	std::cout << double(m) << std::endl;
}
```

