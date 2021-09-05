# Skeletal Animation example from <https://learnopengl.com/Guest-Articles/2020/Skeletal-Animation>.

![Peasant Girl from mixamo.com](Animation.gif)

# How to run.

```
:: Any model with format ASSIMP supports can be specified.
main.exe vampire/dancing_vampire.dae
```

# How to build.

See dependencies from `#include`(s):

 - ASSIMP: `vcpkg install assimp`.
 - GLM: `vcpkg install glm`.
 - STB: `vcpkg install stb`.
 - GLAD: `vcpkg install glad`.
 - GLFW: `vcpkg install glfw3`.

Once installed, compile and link with no special options.
