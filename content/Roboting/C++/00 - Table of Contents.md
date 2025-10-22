# C++ Programming - Index

## Overview
This collection covers essential C++ programming concepts with a focus on modern C++ practices, object-oriented programming "knobs," and practical implementation patterns. The content emphasizes understanding the mechanisms behind C++ features and their proper usage in robotics and systems programming. 

You might be wondering why I learned this. Well I like roboting. [[Why do I keep Roboting?]]

## Topic Structure

### Core Concepts

#### 1. [Passing Arguments](./Passing%20Arguments.md)
**Core concepts:** Parameter passing mechanisms
- **Pass by Value**: Copy semantics
- **Pass by Pointer**: Address copying and ownership implications
- **Pass by Reference**: Alias creation and performance considerations
- Memory and performance implications of each approach

### Object-Oriented Programming Fundamentals

#### 2. [Encapsulation](./Encapsulation.md)
**Core concepts:** Access control and data hiding
- **Access Specifiers**: `public`, `private`, `protected`
- **Friend Classes/Functions**: Breaking encapsulation selectively
- **Getters/Setters**: Controlled access patterns
- **Namespaces**: Code organization and visibility control

#### 3. [Construction and Lifetime](./Construction%20and%20Lifetime.md)
**Core concepts:** Object lifecycle management
- Constructor patterns and initialization
- Destructor behavior and cleanup
- RAII (Resource Acquisition Is Initialization)
- Object lifetime considerations

#### 4. [Inheritance](./Inheritance.md)
**Core concepts:** Class hierarchies and relationships
- Base and derived class relationships
- Virtual functions and method overriding
- Access control in inheritance hierarchies
- Multiple inheritance considerations

#### 5. [Composition, Aggregation & Dependency Injection](./Composition,%20Knobs%20Aggregation%20&%20Dependency%20Injection.md)
**Core concepts:** Object composition patterns
- **Composition**: "Has-a" relationships with strong ownership
- **Aggregation**: "Uses-a" relationships with weak ownership
- **Dependency Injection**: Loose coupling through external dependency provision
- Design patterns for flexible object relationships

### Polymorphism and Advanced OOP

#### 6. [Polymorphism](./Polymorphism.md)
**Core concepts:** "Many forms" - interface flexibility
- **Compile-time Polymorphism**: Function/operator overloading, templates
- **Runtime Polymorphism**: Virtual functions and dynamic dispatch
- **V-tables**: Virtual table mechanism and function pointer resolution
- Virtual destructors and proper cleanup

#### 7. [More Polymorphism](./More%20Polymorphism.md)
**Core concepts:** Polymorphism implementation details
- Specific polymorphism patterns and "knobs"
- Advanced polymorphism techniques
- Performance considerations

### Templates and Generic Programming

#### 8. [Templates](./Templates.md)
**Core concepts:** Compile-time code generation
- **Function Templates**: Generic function implementations
- **Class Templates**: Generic class definitions
- **Non-type Template Parameters**: Compile-time constants
- Template instantiation and specialization

### Language Features and Utilities

#### 9. [explicit keyword](./explicit%20keyword.md)
**Core concepts:** Preventing implicit conversions
- Constructor explicit marking
- Preventing unintended type conversions
- Safer API design patterns

#### 10. [Koenig Lookup](./Koenig%20Lookup.md)
**Core concepts:** Argument-dependent lookup (ADL)
- Name resolution in C++
- Function lookup rules
- Namespace interaction patterns

#### 11. [C Preprocessor Directives](./C%20Preproccessor%20Directives.md)
**Core concepts:** Compile-time text processing
- Macro definitions and usage
- Include guards and header management
- Conditional compilation

### Design Patterns and Best Practices

#### 12. [PIMPL and Forward Declarations](PIMPL%20and%20Forward%20Declarations.md)
**Core concepts:** "Pointer to Implementation" pattern
- Header dependency reduction
- Compilation firewall technique
- Forward declarations and opaque pointers
- Library interface design

#### 13. [Std Library Stuff](./Std%20Library%20Stuff.md)
**Core concepts:** Standard library utilities
- **Move Semantics**: `std::move` and efficient resource transfer
- **Containers**: `std::vector` usage patterns and best practices
- **Algorithms**: `std::copy`, iteration, and algorithm integration
- **Memory Management**: Smart pointers and RAII patterns