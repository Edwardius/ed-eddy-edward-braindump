#polymorphism

Means "many forms", and refers to how a single interface can have many different implementations and handle different underlying types.

## Compile-time Polymorphism (or Static Polymorphism)

Polymorphism that occurs during compile time. The right implementation is chosen by the passed in arguments.

### Function Overloading

The decision to use a specific implementation of a function is made during compilation. Functions can have different implementations based on different arguments. The compiler chooses the right function by matching the usage of the function and its arguments with the right overloaded function.

```c++
class OverloadedClass {
public:
	void print(int A) {
		std::cout << A << std::endl;
	}
	void print(double B) {
		std::cout << B << std::endl;
	}
	void print(std::string S) {
		std::cout << S << std::endl;
	}
};

int main() {
	OverloadedClass overload_class();
	
	double print_this = 100;
	overload_class.print(print_this); // compiler will choose void print(double B); at compile-time
}
```

### Operator Overloading

Similar to function overloading, just with operators.

```c++
class OverloadedOperatorClass {
public:
	OverloadedOperatorClass(int x, int y) : x_(x), y_(y) {}
	
	int get_x() const { return x_; }
	int get_y() const { return y_; }

	OverloadedOperatorClass operator+(const OverloadedOperatorClass& other) {
		return OverloadedOperatorClass(x_ + other.get_x(), y_ + other.get_y());
	}
	
	OverloadedOperatorClass operator-(const OverloadedOperatorClass& other) {
		return OverloadedOperatorClass(x_ - other.get_x(), y_ - other.get_y());
	}
	
private:
	int x_;
	int y_;

};
```

## Runtime Polymorphism (or Dynamic Polymorphism)

The decision to use a specific implementation is decided at runtime (via Vtable lookup). Polymorphism that occurs during runtime. This is done with virtual functions and inheritance. Usually a base class provides some initial interface to override (via virtual methods). **Calls are resolved at runtime using a Virtual Table (vtable) mechanism.**

### Virtual Methods

When a method in a class is set as `virtual`, other classes that inherit it will be able to override the method using the `override` specifier. They can also make they override the be-all-end-all override by specifying `final`

```c++
class BaseShape {
public:
	virtual void draw_shape() { std::cout << "drawing shape\n"; }
	virtual ~BaseShape() = default; // why?
};

class Square : public BaseShape {
public:
	void override draw_shape() final { std::cout << "drawing square\n"; }
};

class Circle : public BaseShape {
public:
	void override draw_shape() final { std::cout << "drawing circle\n"; }
};

class Squares : public Square {
public:
	void override draw_shape() { std::cout << "drawing circle\n"; } // NOT POSSIBLE, because we indicated final on the virtual function override in Square
};
```

### Virtual Destructor

Virtual destructors can be useful for avoiding memory leaks. This is especially important when you are only deleting the base object of a derived class, forgetting to also delete the derived class itself.

```c++
class BaseShape {
public:
	virtual void draw_shape() { std::cout << "drawing shape\n"; }
	~BaseShape() = default; // <-- here I am telling the compiler to use its standard implementation of this destructor function (not virtual)
};

class Star {
public:
	virtual void draw_shape() { std::cout << "drawing star\n" << star_shape << std::endl; }
	~Star() = default;
private:
	std::string star_shape = "_/\_\n";
};

int main() {
	BaseShape* star = new Star();
	delete star; // this will deconstruct BaseShape, but not properly deconstruct Star!!
}

```

Virtual destructor will solve this! The derived class' destructor will be called first before the base class.

```c++
// ################## BETTER WAY ##################
class BaseShape {
public:
	virtual void draw_shape() { std::cout << "drawing shape\n"; }
	virtual ~BaseShape() = default; // <-- here I am telling the compiler to use its standard implementation of this destructor function (not virtual)
};

class Star {
public:
	virtual void draw_shape() { std::cout << "drawing star\n" << star_shape << std::endl; }
	~Star() = default;
private:
	std::string star_shape = "_/\_\n";
};

int main() {
	BaseShape* star = new Star();
	delete star; // this will deconstruct BaseShape base object and Star!
}
```

## V-tables (Virtual Table)

A Virtual Table is a table of function pointers that every class with `virtual` functions spawns with.

For example, given:

```c++
class Base {
public:
	virtual void func1() { std::cout << "base func1\n"; }
	virtual void func2() { std::cout << "base func2\n"; }
};

class Derived : public Base {
public:
	void func1() override { std::cout << "derived func1\n"; }
};
```

The resultant vtables:

```
Base's vtable:
┌──────────────────┐
│ &Base::func1()   │
├──────────────────┤
│ &Base::func2()   │
└──────────────────┘

Derived's vtable:
┌──────────────────────┐
│ &Derived::func1()    │  ← Overridden
├──────────────────────┤
│ &Base::func2()       │  ← Inherited
└──────────────────────┘
```

The mechanism for building vtables is generally as follows:

```
1. Start with Base's vtable layout as a template
2. Copy all entries from Base's vtable
3. For each overridden function in Derived, UPDATE that entry
4. For inherited (non-overridden) functions, KEEP the Base pointer
5. If Derived adds NEW virtual functions, APPEND them to the end
```

This occurs at COMPILE TIME. It also, at a high-level, converts a function call `func1()` to its respective vtable call `call vtable[2]`. At runtime, we just run `call vtable[2]`.