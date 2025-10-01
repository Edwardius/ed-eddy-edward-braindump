#polymorphism

Means "many forms", and refers to how a single interface can have many different implementations and handle different underlying types.

## Compile-time Polymorphism (or Static Polymorphism)

Polymorphism that occurs during compile time. The right implementation is chosen by the passed in arguments.

### Function Overloading

Functions can have different implementations based on different arguments. The compiler chooses the right function by matching the usage of the function and its arguments with the right overloaded function.

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

Polymorphism that occurs during runtime. This is done with virtual functions and inheritance. Usually a base class provides some initial interface to override (via virtual methods). **Calls are resolved at runtime using a Virtual Table (vtable) mechanism.**

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

```