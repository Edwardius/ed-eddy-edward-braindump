#cpp #objectOrientedProgramming #polymorphism 

These refer to specifiers and concepts related to inheritance and how classes can "reuse" or "extend" the functionality from other classes.

**Public/Private/Protected Inheritance:** `public` will keep parent methods / members public when public, protected when protected, private when private. `protected` inheritance will make public things protected. `private` inheritance will make public and protected things private.

**Multiple Inheritance:**  you can compose and share the functionality from multiple bases. You can over overwrite virtual functions from both.

**Final/Override:** `virtual` methods and constructors can be overridden. If an implementation states the overridden method as `final` then no downstream class can override it.

[!tip] If `virtual` is left out, then you lose runtime polymorphism and will resort to the base implementation

### Code Implementation
```c++
// ############## private, public, protected inheritance and virtuals and overrides ##############
class Base {
public:
	void say_public() {
		std::cout << "A\n";
	}
	
	// polymorphic surface
	// const here means "this function promises to not modify the object's state (except for members marked as mutable)"
	virtual void overwrite_me() const {
		std::cout << "I am virtual and I can be overridden.\n";
	}
protected:
	void say_protected() {
		std::cout << "B\n";
	}
private:
	void say_private() {
		std::cout << "C\n";
	}
};

// public inheritance, base's public stays public, protected stays protected, private in unaccesible
class PubChild : public Base {
public:
	void call_base() {
		say_public(); // will work, public interface
		say_protected(); // will work, inherits so protected will be accessible
		say_private(); // will NOT work, private and unaccessible within Base
	}
};

// protected inheritance, base's public AND protected will be become protected to outsiders
class ProtChild : protected Base {
public:
	using Base::say_public; // this will re-expose the public-turned-protected method back as a public

	void call_base() {
		say_public(); // will work, public interface
		say_protected(); // will work, inherits so protected will be accessible
		say_private(); // will NOT work, private and unaccessible within Base
	}
};

// private inheritance, base's public and protected will become private to outsiders
class PrivChild : private Base {
public:
	using Base::say_public; // this will re-expose the public-turned-private method back to public
	
	void call_base() {
		say_public(); // will work, public interface
		say_protected(); // will work, inherits so protected will be accessible
		say_private(); // will NOT work, private and unaccessible within Base
	}
};

// overridding virtual method
class OverridingChild: public Base {
public:
	// the final modifier here means that no further child can override this function, will error out otherwise
	virtual void const override override_me() final {
		std::cout << "I have been overridden\n";
	}
};

// this downstream class will fail to override the virtual method because its final override happenned in OverridingChild
class AnotherOverridingChild: public OverridingChild {
public:
	virtual void const override override_me() {
		std::cout << "I am trying to override\n";
	}
};

// ############## multiple inheritance ##############
class ElectricalButton {
public:
	virtual void push_button() {
		std::cout << "I am pressed the button and it does nothing.\n";
	}
};

class ElectricalLever {
public:
	virtual void flip_lever() {
		std::cout << "I am flipped the lever and it does nothing.\n";
	}
};

class LightFixture : public ElectricalButton, public ElectricalLever {
public:
	virtual void override push_button() {
		std::cout << "I pushed the button and the light turned on.\n";
	}
	
	virtual void override flip_lever() {
		std::cout << "I flipped the lever and the light turned on.\n";
	}
};

// ############## Diamond Problem ##############
class RobotBase {
public:
	RobotBase(int length, int width) {
		length_ = length;
		width_ = width;
	}
private:
	int length_;
	int width_;
};

class TinyRobot : private RobotBase {
public:
	virtual TinyRobot() : RobotBase(0, 0) {}	
};

class BigRobot : private RobotBase {
public:
	virtual BigRobot() : RobotBase(0, 0) {}
};

// Issue is that both TinyRobot and BigRobot inherit from RobotBase, so in order to stop Swarm from initializing two RobotBase, making things ambiguous, we can use virtual specifier to make the most derived class be responsible for constructing the virtual base RobotBase.
class Swarm : public TinyRobot, public BigRobot {
public:
	Swarm(int length, int width) 
		: RobotBase(length, width), TinyRobot(), BigRobot() {}
};
```