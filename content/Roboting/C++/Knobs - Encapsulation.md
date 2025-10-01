#cpp #objectOrientedProgramming

These refer to specifiers and keywords that control what you hide or expose details to downstream objects.

**Access specifiers:** Define what class methods / members are accessible externally.
- `public`: accessible externally
- `private`: only accessible internally, *also with `friends`*
- `protected`:  only accessible internally and to classes that inherit the class, *also with `friends`*

**Getters and Setters:** Exposes read and write functionality to private members through a public interface (that is, a public class method).

**Friend classes and functions:** `friend` specifier gives special access to private / protected members and methods. breaking encapsulation.

**Namespaces:** `namespace` specifier controls visibility of code to others. Code within the same namespace can see each other, while code outside of the namespace has to specify the namespace to properly interface with it.
### Code Implementation

```c++
namespace wato::power_management {

class Battery {
public:
	double get_percent() {
		return percent_;
	}
private:
	double percent_ = 100;
	
	friend class BatteryTesterFriend; // BatteryTesterFriend can see private stuff
	friend std::ostream& operator<<(std::ostream& os, const Battery& B);
};

class BatteryTesterNotFriend {
public:
	bool track_percent(Battery& B) { // this will pass because we are using Battery's public interface
		tracked_percentage_ = B.get_percent();
		return true;
	}
	bool drain(Battery& B) { // this will fail because this class is not a friend of Battery, and thus cannot directly set percent_
		B.percent_ = 0;
		return true;
	}
private:
	double tracked_percentage_;
};

class BatteryTesterFriend {
public:
	bool track_percent(Battery& B) { // this will pass because we are using Battery's public interface
		tracked_percentage_ = B.get_percent();
		return true;
	}
	bool drain(Battery& B) { // this will pass because this class IS a friend of Battery, and thus can directly mutate percent_
		B.percent_ = 0;
		return true;
	}
private:
	double tracked_percentage_;
};

// friend function to print the battery percentage
std::ostream& operator<<(std::ostream& os, const Battery& B) {
	os << "Battery.percent_" << B.percent_;
	return os;
}

} // namespace wato::power_management

int main() {
	wato::power_management::Battery test_battery();
	std::cout << test_battery << std::endl;
}
```

