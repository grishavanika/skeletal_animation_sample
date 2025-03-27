# Skeletal Animation

Example from <https://learnopengl.com/Guest-Articles/2020/Skeletal-Animation>:

![](Capoeira_Mannequin.gif)

# How to run

Use Collada format instead of FBX because the code does not support
formats with embedded textures; expects texture files to be present on disk:

```
build\Release\main.exe Capoeira_Mannequin.dae
```

`Capoeira_Mannequin.dae` comes from [mixamo](https://www.mixamo.com/), select "Mannequin" for Characters and "Capoeira" for Animations. Original dancing_vampire.dae from the article could be on [LearnOpenGL github](https://github.com/JoeyDeVries/LearnOpenGL/tree/6159792dec67ff0ba70f7fd2eafd88b683730e64/resources/objects/vampire).  

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
