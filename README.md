# Skeletal Animation

Example from <https://learnopengl.com/Guest-Articles/2020/Skeletal-Animation>:

![](dancing_vampire.gif)

# How to run

```
main dancing_vampire.dae
```

`dancing_vampire.dae` comes from [LearnOpenGL github](https://github.com/JoeyDeVries/LearnOpenGL/tree/6159792dec67ff0ba70f7fd2eafd88b683730e64/resources/objects/vampire).  

# How to build

Just CMake with VCPKG:

```
cmake -S . -B build ^
	-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release
```

Alternatively, dependencies are (see vcpkg.json):

 - ASSIMP: `vcpkg install assimp`.
 - GLM: `vcpkg install glm`.
 - STB: `vcpkg install stb`.
 - GLAD: `vcpkg install glad`.
 - GLFW: `vcpkg install glfw3`.

Once installed, compile and link main.cpp with C++20 enabled.
