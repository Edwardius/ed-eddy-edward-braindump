#programming #debian #ros2 

# Package mapping with `bloom`
ROS2's build farm, known as `bloom` provides a direct translation between a typical ROS2 package and its debian counterpart.

Given a ros2 package:
```
my_robot_pkg/
├── package.xml       # ROS2 package metadata
├── CMakeLists.txt    # Build instructions
├── src/              # C++ source
├── include/          # Headers
├── launch/           # Launch files
└── config/           # Parameters
```

Bloom does a mapping:
- `package.xml` -> `debian/control`
- `CMakeLists.txt` -> `debian/rules`
and dependencies into their respective debian dependencies

# Using `bloom`

```bash
colcon build...
bloom-generate rosdebian --os-name ubuntu --ros-distro ROS_DISTRO
```

This generates a `debian/` with all the stuff. And that's all folks...

# Creating an APT Repository

Hosting a basic apt repository is not too bad, can be hosted with github pages or directly on ubuntu PPA.


