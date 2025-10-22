#cpp #programming 

These are common C++ idioms that people follow when interfacing classes together.

## Composition

```
+-----------+        owns (strong)
|   Car     |◆───────> +--------+
|           |          | Engine |
+-----------+          +--------+
    |
    | by-value member or unique_ptr member
    v
fields: Engine e_;   // or: std::unique_ptr<Engine> e_;
```

The idea here is that the whole owns the part. That is, a derived class contains the object directly as a data member, or as a `unique_ptr`.

```c++
class Engine {
public:
	explicit Engine(int horsepower) : hp_(horsepower) {}
	void start() const noexcept {
		std::cout << "vroom\n";
	}
private:
	int hp_;
};

class Car {
public:
	explicit Car(int horsepower) : engine_(horsepower) {}
	void start_car() const {
		engine_.start();
		std::cout << "car has started!\n";
	}
private:
	Engine engine_
};

// You can also do something similar using a unique_ptr
class CarUnique {
public:
	explicit Car(int horsepower) : engine_(std::make_unique<Engine>(horsepower)) {}
	void start_car() const {
		engine_->start();
		std::cout << "car has started!\n";
	}
private:
	std::unique_ptr<Engine> engine_;
};
```
## Aggregation

```
+-----------+       non-owning raw ptr        +-----------+
|   Team    | ------------------------------> |  Player   |
+-----------+                                 +-----------+
```

The idea here is that the object has reference to a part it needs, but does not own the part itself. This is to avoid accidental lifetime extensions.

```c++
class Player {
public:
	Player(std::string name) : name_(name) {}
	
	void intro() { std::cout << "I am" << name << std::endl; }
private:
	std::string name_;
};

class Team {
public:
	Team() = default;
	
	void add_player(Player* p) {
		players.push_back(p);
	}
	void roll_call() {
		for (auto* p: players_) p->intro();
	}
private:
	std::vector<Player*> players;
}

// can also do a similar thing with weak_ptrs
class TeamWeakPtr {
public:
	Team() = default;
	
	void add_player(const std::shared_ptr<Player>& p) {
		players.push_back(p); // this implicitly will store weak_ptrs
	}
	void roll_call() {
		for (const auto& p : players_) {
			if (auto p_ptr = p.lock()) { p_ptr->intro(); }
		}
	}
private:
	std::vector<std::weak_ptr<Player>> players;
}

int main() {
	Team team();
	TeamWeakPtr team_weak_ptr();
	
	// for regular
	Player alice("alice");
	Player bob("bob");
	team.add_player(alice);
	team.add_player(bob);
	team.roll_call();
	
	// for weak_ptr
	auto alice = std::make_shared<Player>("alice");
	auto bob = std::make_shared<Player>("bob");
	team.add_player(alice);
	team.add_player(bob);
	team.roll_call();
}
```

> **Aggregation vs Composition** the main difference is the lifetime management. Does the object and its component cleanup at the same time? ie. closely locked in ownership, **then it is composition**. Can the component stay alive without the object? ie. they are loosely connected in ownership, **then is is aggregation**.
> 
> **Aggregation** -> reference to the component object, can stay alive when object dies
> **Composition** -> lifetime of component is decided by the lifetime of the object
## Dependency Injection

```
                 +-------------------+
                 |   Storage (IF)    |<-- virtual API
                 +-------------------+
                  ^               ^
implements        |               |
+-----------------+---+       +---+-----------------+
| InMemoryStorage     |       | SqlStorage          |
+---------------------+       +---------------------+

                   injected (owns or borrows)
+--------------------+
|     Repository     |----->[ Storage, either InMemoryStorage or SqlStorage ]  (unique_ptr or reference)
+--------------------+
```

This is the idea of having a class accept a pre-built object. This differs for composition because the object it depends on is initialized in its constructor.
#### Example using Static [[Polymorphism]]

```c++
class MapStorage {
    std::map<std::string, std::string> kv_;
public:
    void put(const std::string& k, std::string v) { kv_[k]=std::move(v); }
    std::string get(const std::string& k) const {
        auto it = kv_.find(k); return (it==kv_.end()) ? "" : it->second;
    }
};

template <class StoragePolicy>
class RepoT {
    StoragePolicy storage_;                          // value or reference wrapper
public:
    RepoT() = default;
    explicit RepoT(StoragePolicy s) : storage_(std::move(s)) {}
    void save_user(const std::string& id, std::string name) {
        storage_.put("user:"+id, std::move(name));
    }
    std::string load_user(const std::string& id) const {
        return storage_.get("user:"+id);
    }
};

int main() {
    RepoT<MapStorage> repo;                          // choose policy at compile time
    repo.save_user("7", "Alice");
    std::cout << repo.load_user("7") << "\n";
}
```

You used this in your templated test classes! To make base test nodes that handle any ros publisher handling any sort of message type!

#### Example using Dynamic [[Polymorphism]]

```c++
class Storage {
public:
    virtual ~Storage() = default;
    virtual void put(const std::string&, const std::string&) = 0;
    virtual std::string get(const std::string&) const = 0;
};

// I can implement different storage options and make repository use any of them!
class InMemoryStorage : public Storage {
    std::map<std::string, std::string> kv_;
public:
    void put(const std::string& k, const std::string& v) override { kv_[k]=v; }
    std::string get(const std::string& k) const override {
        auto it = kv_.find(k); return (it==kv_.end()) ? "" : it->second;
    }
};

class Repository {
    std::unique_ptr<Storage> storage_; // HERE WE ARE POINTING TO THE BASE OBJECT!!
public:
    explicit Repository(std::unique_ptr<Storage> s) : storage_(std::move(s)) {}
    void save_user(const std::string& id, const std::string& name) {
        storage_->put("user:"+id, name);
    }
    std::string load_user(const std::string& id) const {
        return storage_->get("user:"+id);
    }
};

int main() {
	auto mem_storage = std::make_unique<InMemoryStorage>();
    Repository repo{std::move(mem_storage)};   // inject impl
    repo.save_user("88", "Eddy");
    std::cout << repo.load_user("88") << "\n";
}
```

## Factory

Generally refers to the idea of having a class or function be responsible for producing the right object with the right internal composition for the end user of the library based on some requirements.

### Simple Factory

Most common, accepts parameters and builds an object accordingly.

```c++
class Renderer {
public:
    virtual ~Renderer() = default;
    virtual void draw() const = 0;
};

class OpenGLRenderer : public Renderer {
public:
    void draw() const override { std::cout << "OpenGL draw\n"; }
};

class VulkanRenderer : public Renderer {
public:
    void draw() const override { std::cout << "Vulkan draw\n"; }
};

class RendererFactory {
public:
    static std::unique_ptr<Renderer> create(const std::string& api) {
        if (api == "opengl") return std::make_unique<OpenGLRenderer>();
        if (api == "vulkan") return std::make_unique<VulkanRenderer>();
        throw std::invalid_argument("unknown api");
    }
};

int main() {
    auto r = RendererFactory::create("vulkan");
    r->draw();
}
```

### Factory Method

No central switch like the simple factory. Instead builds based on defined subclasses using the things you want to use. More of a complete composer of parts.

```c++
class Renderer {
public:
    virtual ~Renderer() = default;
    virtual void draw() const = 0;
};
class OpenGLRenderer : public Renderer { public: void draw() const override { std::cout << "OpenGL\n"; } };
class VulkanRenderer : public Renderer { public: void draw() const override { std::cout << "Vulkan\n"; } };

class App {
public:
    virtual ~App() = default;
    void run() const { auto r = create_renderer(); r->draw(); }
private:
    virtual std::unique_ptr<Renderer> create_renderer() const = 0; // factory method
};

class GameApp : public App {
private:
    std::unique_ptr<Renderer> create_renderer() const override {
        return std::make_unique<OpenGLRenderer>();
    }
};

class CadApp : public App {
private:
    std::unique_ptr<Renderer> create_renderer() const override {
        return std::make_unique<VulkanRenderer>();
    }
};

int main() {
    std::unique_ptr<App> app = std::make_unique<GameApp>();
    app->run();
}
```
### Abstract Factory

More of a grouping pattern. Creates families that must match.

```c++
class Button { public: virtual ~Button() = default; virtual void paint() const = 0; };
class Checkbox { public: virtual ~Checkbox() = default; virtual void paint() const = 0; };

class DarkButton : public Button { public: void paint() const override { std::cout << "Dark Button\n"; } };
class DarkCheckbox : public Checkbox { public: void paint() const override { std::cout << "Dark Checkbox\n"; } };
class LightButton : public Button { public: void paint() const override { std::cout << "Light Button\n"; } };
class LightCheckbox : public Checkbox { public: void paint() const override { std::cout << "Light Checkbox\n"; } };

class WidgetFactory {
public:
    virtual ~WidgetFactory() = default;
    virtual std::unique_ptr<Button> create_button() const = 0;
    virtual std::unique_ptr<Checkbox> create_checkbox() const = 0;
};

class DarkFactory : public WidgetFactory {
public:
    std::unique_ptr<Button> create_button() const override { return std::make_unique<DarkButton>(); }
    std::unique_ptr<Checkbox> create_checkbox() const override { return std::make_unique<DarkCheckbox>(); }
};

class LightFactory : public WidgetFactory {
public:
    std::unique_ptr<Button> create_button() const override { return std::make_unique<LightButton>(); }
    std::unique_ptr<Checkbox> create_checkbox() const override { return std::make_unique<LightCheckbox>(); }
};

int main() {
    std::unique_ptr<WidgetFactory> f = std::make_unique<DarkFactory>();
    auto btn = f->create_button();
    auto cb  = f->create_checkbox();
    btn->paint(); cb->paint();
}
```
