#cpp #programming 

These are concepts and specifiers that deal with construction, lifetime, and memory management. 

## RAII (Resource Acquisition is Initialization)

This is a C++ idiom that everyone should follow. It states that the **lifetime of a resource should be tied to the lifetime of the object**. This is to generally stop memory leaks. There are various ways to manage this:
## Rule of 5

A more explicit way of managing RAII.

*If your C++ class manages a resource (not implicitly managed memory, file handle, socket, mutex, etc.), and you need to customize any one of its special member functions, you probably need to define (or explicitly call `= default` / `= delete`) all five:*

1. Destructor
2. Copy Constructor
3. Copy Assignment
4. Move Constructor
5. Move Assignment

Reason why is to avoid misusage and memory leaks that can occur from implicit operations.

```c++
class Buffer {
public:
	// Constructors ( not part of 5, but still needed regardless )
	Buffer() = default;
	explicit Buffer(std::size_t n) 
	: n_(n), data_(n? new int[n]{} : nullptr) {} // creating a raw array int* data_ = new int[n]{} <--- makes an array with 0 initialized as all elements

	// Destructor
	~Buffer() {
		delete[] data_;
	}

	// Copy Constructor (deep copy)
	// we are allowed to access other's private members BECAUSE THEY ARE BOTH OF THE SAME CLASS
	Buffer(const Buffer& other)
	: n_(other.n_), data_(other.data_ ? new int[other.n_] : nullptr) {
		std::copy_n(other.data_, n_, data_);
	}
	
	// Copy Assignment (deep-copy into a temp and then swap)
	Buffer& operator=(const Buffer& other) {
		if (this != &other) {
			Buffer temp(other);
			swap(temp);
		}
		return this*;
	}

	// Move Constructor
	Buffer(Buffer&& other) noexcept
	: n_(other.n_), data_(other.data_) {
		other.n_ = 0; 
		other.data_ = nullptr; // <-- this stops the old data from deleting the new data (double delete)
	}

	// Move Assignment
	Buffer& operator=(Buffer&& other) noexcept {
		if (this != &other) { // <-- self assignment guard, ensures that we are not trying to move assignment the same object in memory
			delete[] data_;
			n_ = other.n_;
			data_ = other.data_;
			
			// we want to stop the chancs of a double delete
			other.n_ = 0;
			other.data_ = nullptr;
		}
		return this*;
	}
	
	// Helper methods
	void fill(int v) {
		for (std::size_t i = 0; i < n_; i++) {
			data_[i] = v;
		}
	}

	void swap(Buffer& rhs) noexcept {
		std::swap(n_, rhs_.n_);
		std::swap(data_, rhs_.data_);
	}
	// Usage of friend here is interesting, its to make it work with std::swap becuase of ADL (Argument Dependant Lookup, Koenig Lookup)
	friend void swap(Buffer& a, Buffer& b) noexcept { a.swap(b); }

private:
	std::size_t n_ = 0;
	int* data_ = nullptr;
};

int main() {
	// Use of constructor
	Buffer src1(3); src1.fill(11);
	Buffer src2(4); src1.fill(22);
	Buffer src3(5); src1.fill(33);
	Buffer src4(6); src1.fill(44);
	
	// Use copy constructor (creates a new Buffer with deep-copy)
	Buffer copyConstructor = src1;

	// Use of copy assignment (overwrites existing Buffer's content)
	Buffer copyAssignment(1);
	copyAssignment = src2;

	// Use of move constructor (takes ownership of the contents of src3)
	Buffer moveConstructor = std::move(src3);

	// Use of move assignment 
	Buffer moveAssignment(3);
	moveAssignment = std::move(src4);
}

```

For extras in the code block above, see [[Koenig Lookup]] and [[Passing Arguments]]

## Smart Pointers 

Smart pointers are, in my opinion, a cleaner way to deal with RAII. They are "smarter" than regular pointers because they can help manage memory and ownership. They also reduce the need to deal with the Rule of 5 and managing memory with `new` and `delete`.

Types of Smart Pointers:
- `std::unique_ptr<T>`  sole owner (move-only)
- `std::shared_ptr<T>` shared owner (has a reference count)
- `std::weak_ptr<T>` observer (non-owning handle to a shared_ptr. Loses access once out of scope.)
### Unique Pointer
`unique_ptr` is for objects that should have a single owner at all times. It is move-only, so you should be transferring objects with `std::move`. 

**Great for:** [[Composition, Knobs Aggregation & Dependency Injection|Composition]], [[PIMPL]], Containers, Factories

**Key Points:**
- Moveable but **not copyable**
- Can pass in a **custom deleter** to let the smart pointer know how to properly cleanup resources
	- usually is a workaround, you generally won't need to specify a custom deleter if the object you are pointing to has a proper destructor.
	- this is usually used with incomplete types and implementations like PIMPL

```c++
#include <memory> <-- for smart pointers

class Engine {
public: 
	void start_engine() const {
		std::cout << "vroom\n";
	}
};

class Car {
public:
	explicit Car(std::unique_ptr<Engine> engine) 
	: engine_(std::move(engine) {}

	void start_car() const {
		engine_->start_engine();
		std::cout << "driving\n";
	}
private:
	std::unique_ptr<Engine> engine_;
};

// Factory to create car
// std::make_unique<T>(T args...)
std::unique_ptr<Car> make_car() {
	return std::make_unique<Car>(std::make_unique<Engine>())
}

int main() {
	auto car = make_car();
	car->drive();
	
	// This moves the car into the garage
	// ie. car is empty and garage[0] now has the car
	std::vector<std::unique_ptr<Car>> garage;
	garage.push_back(std::move(car));
}
```

### Shared Pointer
`shared_ptr` is for objects that could have multple owners. **last shared_ptr that goes out of scope, or is destroyed, actually deletes the object its pointing to (because ref_count goes to 0.**

**Use when:** parts of your program need to own the same object's lifetime. Good to **avoid by default**... but rclcpp uses this pretty often lol.

**Key Points:**
- copying increases the ref-count, destruction decreases the ref-count. The last shared_ptr deletes the object it's pointing to.
- `enable_shared_from_this<T>` lets you make a class safely create a `shared_ptr` to itself. This is useful when you wanna run async calls or want to return a shared_ptr to your object for some reason.

```c++
// enable_shared_from_this is a curiously recurring template pattern!
class Task : public std::enable_shared_from_this<Task> {
public:
    void start_async() {
        auto self = shared_from_this();          // keep alive during async work
        std::thread([self]{ // <-- here! we pass a shared pointer of itself int he thread to keep it alive!
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            std::cout << "Task still alive: " << self.use_count() << " owners\n";
        }).detach();
    }
};

int main() {
	auto t = std::make_shared<Task>();
	t->start_async();
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
```

#### Why does rclcpp use shared_ptr's so much?
ROS constructs run inside an executor that deals with a bunch of async calls. Thus theres a strong usage of shared_ptr to keep them all alive. The use of ref_count also helps with destruction ordering in the underlying rcl libraries.
### Weak Pointer
`weak_ptr` is a non-owning observer of a `shared_ptr`. It gets permission to read-write to an object owned by a `shared_ptr` within a scope.

**Use when:** You want to manage a `shared_ptr`-accessed object **without extending its lifetime.**

**Key Points:**
- `weak_ptr` needs to obtain a lock on the `shared_ptr`'s object before accessing it.
- `std::weak_ptr<T>` can mutate T if given lock
- `std::weak_ptr<const T>` can only read T if given the lock

```c++
// Singly Linked List
class Node : public std::enable_shared_from_this<Node> {
public:
	explicit Node(std::string name)
	: name_(std::move(name)) {}
	
	void set_child(const std::shared_ptr<Node> child) {
		child_ = child;
		parent_ = shared_from_this();
	}
	
	void print_chain_up() const {
		std::cout << name_;
		if (auto p = parent_.lock()) { // checks if parent is alive
			std::cout << " <- ";
			p->print_chain_up();
		} else {
			std::cout << " <- [root]\n ";
		}
	}
private:
	std::string name_;
	std::shared_ptr<Node> child_;
	std::weak_ptr<Node> parent_;
};

int main() {
	auto root = std::make_shared<Node>("root");
	auto leaf = std::make_shared<Node>("leaf");
	
	root->set_child(leaf);
	leaf->print_chain_up(); // will print leaf <- root <- [root]
}
```

