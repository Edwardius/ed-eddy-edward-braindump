A module is a collection of items: functions, structs, trait, `impl` blocks, and other modules.

Lets us split up code into logical units and manage visibility between them.

## Visibility
By default, items in a module have private visibility. Can be overridden with the `pub` modifier. Only public items can be access from outside the module scope.

```rust
mod my_mod {
	fn private_function() {
		println!("called private function");
	}
	
	pub fn public_function() {
		println!("called public function");
	}
	
	pub fn indirect_access() {
		private_function(); // <--- we can access private items through a public interface!
	}
	
	pub mod nested {
		pub fn function() {
			println!("called my_mod::nested::function");
		}
		
		fn priv_function() {
			println!("called my_mod::nested::priv_function");
		}
		
		// only visible/public from a given path (so from my_mod)
		pub(in crate::my_mod) fn public_function_in_my_mod() {
			public_function_in_nested();
		}
		
		// only visible within the current module, same as leaving private
		pub(self) fn public_function_in_nested(){
            println!("called `my_mod::nested::public_function_in_nested()`");
		}
		
		// only visible within theparent module
		pub(super) fn public_function_in_super_mod() {
            println!("called `my_mod::nested::public_function_in_super_mod()`");
        }
	}
	
    pub fn call_public_function_in_my_mod() {
        print!("called `my_mod::call_public_function_in_my_mod()`, that\n> ");
        nested::public_function_in_my_mod();
        print!("> ");
        nested::public_function_in_super_mod();
    }

    // pub(crate) makes functions visible only within the current crate
    pub(crate) fn public_function_in_crate() {
        println!("called `my_mod::public_function_in_crate()`");
    }

    // Nested modules follow the same rules for visibility
    mod private_nested {
        #[allow(dead_code)]
        pub fn function() {
            println!("called `my_mod::private_nested::function()`");
        }

        // Private parent items will still restrict the visibility of a child item,
        // even if it is declared as visible within a bigger scope.
        #[allow(dead_code)]
        pub(crate) fn restricted_function() {
            println!("called `my_mod::private_nested::restricted_function()`");
        }
    }
    
}
```

## Struct Visibility
Defaults of private and can be overridden with a pub modifier. 

```rust
mod my {
    // A public struct with a public field of generic type `T`
    pub struct OpenBox<T> {
        pub contents: T,
    }

    // A public struct with a private field of generic type `T`
    pub struct ClosedBox<T> {
        contents: T,
    }

    impl<T> ClosedBox<T> {
        // A public constructor method
        pub fn new(contents: T) -> ClosedBox<T> {
            ClosedBox {
                contents: contents,
            }
        }
    }
}

fn main() {
    // Public structs with public fields can be constructed as usual
    let open_box = my::OpenBox { contents: "public information" };

    // and their fields can be normally accessed.
    println!("The open box contains: {}", open_box.contents);

    // Public structs with private fields cannot be constructed using field names.
    // Error! `ClosedBox` has private fields
    //let closed_box = my::ClosedBox { contents: "classified information" };

    // However, structs with private fields can be created using
    // public constructors
    let _closed_box = my::ClosedBox::new("classified information");

    // and the private fields of a public struct cannot be accessed.
    // Error! The `contents` field is private
    //println!("The closed box contains: {}", _closed_box.contents);
}
```

## Use Declaration
Its the same as using namespace std; in cpp. It shortens the path to a new name to mak eur life easier.
```rust
use crate::deeply::nested::{
	my_func,
}

fn main() {
	my_func();
}
```

Can also use the as keyword to bind to a different name
```rust 
use deeply::nested::function as other_function;

fn main() {
	other_function();
}
```

## Super and Self
`super` and `self` can be used to specify which function you are referring to. either the function inside the same scope as your module (self), or outside the modules (super)

```rust
fn function() {
    println!("called `function()`");
}

mod cool {
    pub fn function() {
        println!("called `cool::function()`");
    }
}

mod my {
    fn function() {
        println!("called `my::function()`");
    }

    mod cool {
        pub fn function() {
            println!("called `my::cool::function()`");
        }
    }

    pub fn indirect_call() {
        // Let's access all the functions named `function` from this scope!
        print!("called `my::indirect_call()`, that\n> ");

        // The `self` keyword refers to the current module scope - in this case `my`.
        // Calling `self::function()` and calling `function()` directly both give
        // the same result, because they refer to the same function.
        self::function();
        function();

        // We can also use `self` to access another module inside `my`:
        self::cool::function();

        // The `super` keyword refers to the parent scope (outside the `my` module).
        super::function();

        // This will bind to the `cool::function` in the *crate* scope.
        // In this case the crate scope is the outermost scope.
        {
            use crate::cool::function as root_function;
            root_function();
        }
    }
}

fn main() {
    my::indirect_call();
}
```

## File Hierarchy
Module can be mapped to a file/directory hierarchy. 

Given we have 
```
$ tree .
.
├── my
│   ├── inaccessible.rs
│   └── nested.rs
├── my.rs
└── split.rs
```

`my.rs `contains the code for the `my` module itself
`my/ `contains files for any submodules declared inside `my.rs`

